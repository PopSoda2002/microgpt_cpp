#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>

// ─────────────────────────────────────────
//  Autograd Value  (shared_ptr managed)
// ─────────────────────────────────────────
struct Value;
using SP = std::shared_ptr<Value>;

struct Value {
    double data, grad;
    std::vector<SP>     children;
    std::vector<double> local_grads;
    Value(double d) : data(d), grad(0) {}
};

SP make_val(double d, std::vector<SP> ch = {}, std::vector<double> lg = {}) {
    auto v = std::make_shared<Value>(d);
    v->children    = std::move(ch);
    v->local_grads = std::move(lg);
    return v;
}

SP vadd(SP a, SP b)     { return make_val(a->data + b->data, {a,b}, {1.0,1.0}); }
SP vmul(SP a, SP b)     { return make_val(a->data * b->data, {a,b}, {b->data, a->data}); }
SP vpow(SP a, double e) { return make_val(std::pow(a->data,e), {a}, {e*std::pow(a->data,e-1)}); }
SP vlog(SP a)           { return make_val(std::log(a->data), {a}, {1.0/a->data}); }
SP vexp(SP a)           { double ex=std::exp(a->data); return make_val(ex,{a},{ex}); }
SP vrelu(SP a)          { return make_val(std::max(0.0,a->data),{a},{a->data>0?1.0:0.0}); }
SP vneg(SP a)           { return make_val(-a->data,{a},{-1.0}); }
SP vsub(SP a, SP b)     { return vadd(a,vneg(b)); }
SP vdiv(SP a, SP b)     { return vmul(a,vpow(b,-1.0)); }
SP vscale(SP a,double s){ return make_val(a->data*s,{a},{s}); }
SP vcst(double d)       { return make_val(d); }

void backward(SP root) {
    std::vector<SP> topo;
    std::set<Value*> visited;
    std::function<void(SP)> build = [&](SP v){
        if(!visited.count(v.get())){
            visited.insert(v.get());
            for(auto& c:v->children) build(c);
            topo.push_back(v);
        }
    };
    build(root);
    root->grad = 1.0;
    for(int i=(int)topo.size()-1;i>=0;--i){
        auto& v=topo[i];
        for(int j=0;j<(int)v->children.size();++j)
            v->children[j]->grad += v->grad * v->local_grads[j];
    }
}

// ─────────────────────────────────────────
//  Types & random
// ─────────────────────────────────────────
using Vec = std::vector<SP>;
using Mat = std::vector<Vec>;

std::mt19937 rng(42);
std::normal_distribution<double> gauss(0.0,0.08);

Mat make_matrix(int nout,int nin){
    Mat m(nout,Vec(nin));
    for(auto& row:m) for(auto& p:row) p=make_val(gauss(rng));
    return m;
}

// ─────────────────────────────────────────
//  NN ops
// ─────────────────────────────────────────
Vec linear(const Vec& x,const Mat& w){
    Vec out; out.reserve(w.size());
    for(auto& row:w){
        SP acc=vcst(0.0);
        for(int i=0;i<(int)x.size();++i) acc=vadd(acc,vmul(row[i],x[i]));
        out.push_back(acc);
    }
    return out;
}

Vec softmax(const Vec& logits){
    double mv=logits[0]->data;
    for(auto& v:logits) mv=std::max(mv,v->data);
    Vec exps; for(auto& v:logits) exps.push_back(vexp(vsub(v,vcst(mv))));
    SP se=vcst(0.0); for(auto& e:exps) se=vadd(se,e);
    Vec out; for(auto& e:exps) out.push_back(vdiv(e,se));
    return out;
}

Vec rmsnorm(const Vec& x){
    SP ms=vcst(0.0);
    for(auto& xi:x) ms=vadd(ms,vmul(xi,xi));
    ms=vscale(ms,1.0/x.size());
    SP sc=vpow(vadd(ms,vcst(1e-5)),-0.5);
    Vec out; for(auto& xi:x) out.push_back(vmul(sc,xi));
    return out;
}

// ─────────────────────────────────────────
//  Hyperparams & model
// ─────────────────────────────────────────
const int N_LAYER=1, N_EMBED=16, SLIDING_WIN=16, N_HEAD=4, HEAD_DIM=N_EMBED/N_HEAD;
std::map<std::string,Mat> state_dict;
int vocab_size, BOS_ID;

void init_model(){
    state_dict["wte"]    =make_matrix(vocab_size,N_EMBED);
    state_dict["wpe"]    =make_matrix(SLIDING_WIN,N_EMBED);
    state_dict["lm_head"]=make_matrix(vocab_size,N_EMBED);
    for(int i=0;i<N_LAYER;++i){
        std::string s=std::to_string(i);
        state_dict["layer"+s+".attn_wq"]=make_matrix(N_EMBED,N_EMBED);
        state_dict["layer"+s+".attn_wk"]=make_matrix(N_EMBED,N_EMBED);
        state_dict["layer"+s+".attn_wv"]=make_matrix(N_EMBED,N_EMBED);
        state_dict["layer"+s+".attn_wo"]=make_matrix(N_EMBED,N_EMBED);
        state_dict["layer"+s+".mlp_fc1"]=make_matrix(N_EMBED*4,N_EMBED);
        state_dict["layer"+s+".mlp_fc2"]=make_matrix(N_EMBED,N_EMBED*4);
    }
}

std::vector<SP> all_params(){
    std::vector<SP> ps;
    for(auto& [_,mat]:state_dict)
        for(auto& row:mat) for(auto& p:row) ps.push_back(p);
    return ps;
}

// ─────────────────────────────────────────
//  Forward
// ─────────────────────────────────────────
Vec gpt_forward(int token_id,int pos_id,
                std::vector<std::vector<Vec>>& keys,
                std::vector<std::vector<Vec>>& vals){
    Vec x(N_EMBED);
    for(int i=0;i<N_EMBED;++i)
        x[i]=vadd(state_dict["wte"][token_id][i],state_dict["wpe"][pos_id][i]);
    x=rmsnorm(x);

    for(int li=0;li<N_LAYER;++li){
        std::string s=std::to_string(li);
        Vec xr=x; x=rmsnorm(x);
        Vec q=linear(x,state_dict["layer"+s+".attn_wq"]);
        Vec k=linear(x,state_dict["layer"+s+".attn_wk"]);
        Vec v=linear(x,state_dict["layer"+s+".attn_wv"]);
        keys[li].push_back(k); vals[li].push_back(v);

        Vec x_attn;
        double sc=1.0/std::sqrt((double)HEAD_DIM);
        for(int h=0;h<N_HEAD;++h){
            int hs=h*HEAD_DIM;
            Vec q_h(q.begin()+hs,q.begin()+hs+HEAD_DIM);
            int T=(int)keys[li].size();
            Vec al;
            for(int t=0;t<T;++t){
                SP dot=vcst(0.0);
                for(int j=0;j<HEAD_DIM;++j) dot=vadd(dot,vmul(q_h[j],keys[li][t][hs+j]));
                al.push_back(vscale(dot,sc));
            }
            Vec aw=softmax(al);
            for(int j=0;j<HEAD_DIM;++j){
                SP acc=vcst(0.0);
                for(int t=0;t<T;++t) acc=vadd(acc,vmul(aw[t],vals[li][t][hs+j]));
                x_attn.push_back(acc);
            }
        }
        x=linear(x_attn,state_dict["layer"+s+".attn_wo"]);
        for(int i=0;i<N_EMBED;++i) x[i]=vadd(x[i],xr[i]);

        xr=x; x=rmsnorm(x);
        x=linear(x,state_dict["layer"+s+".mlp_fc1"]);
        for(auto& xi:x) xi=vrelu(xi);
        x=linear(x,state_dict["layer"+s+".mlp_fc2"]);
        for(int i=0;i<N_EMBED;++i) x[i]=vadd(x[i],xr[i]);
    }
    return linear(x,state_dict["lm_head"]);
}

// ─────────────────────────────────────────
//  Main
// ─────────────────────────────────────────
int main(){
    std::ifstream fin("input.txt");
    if(!fin){std::cerr<<"input.txt not found\n";return 1;}
    std::vector<std::string> docs;
    std::string line;
    while(std::getline(fin,line)) if(!line.empty()) docs.push_back(line);
    std::shuffle(docs.begin(),docs.end(),rng);
    std::cout<<"num docs: "<<docs.size()<<"\n";

    std::string ac; for(auto& d:docs) ac+=d;
    std::set<char> cs(ac.begin(),ac.end());
    std::vector<char> uchars(cs.begin(),cs.end());
    std::sort(uchars.begin(),uchars.end());
    BOS_ID=uchars.size(); vocab_size=BOS_ID+1;
    std::cout<<"BOS: "<<BOS_ID<<", vocab_size: "<<vocab_size<<"\n";

    auto c2id=[&](char c){return(int)(std::find(uchars.begin(),uchars.end(),c)-uchars.begin());};

    init_model();
    auto params=all_params();
    std::cout<<"num params: "<<params.size()<<"\n";

    double lr=0.01,beta1=0.85,beta2=0.99,eps=1e-8;
    std::vector<double> m_adam(params.size(),0.0),v_adam(params.size(),0.0);

    int num_steps=2000;
    for(int step=0;step<num_steps;++step){
        auto& doc=docs[step%docs.size()];
        std::vector<int> tokens={BOS_ID};
        for(char c:doc) tokens.push_back(c2id(c));
        tokens.push_back(BOS_ID);
        int n=std::min(SLIDING_WIN,(int)tokens.size()-1);

        std::vector<std::vector<Vec>> keys(N_LAYER),vals(N_LAYER);
        SP loss=vcst(0.0);
        for(int pos=0;pos<n;++pos){
            Vec logits=gpt_forward(tokens[pos],pos,keys,vals);
            Vec probs=softmax(logits);
            loss=vadd(loss,vneg(vlog(probs[tokens[pos+1]])));
        }
        loss=vscale(loss,1.0/n);
        backward(loss);

        double lr_t=lr*(1.0-(double)step/num_steps);
        for(int i=0;i<(int)params.size();++i){
            double g=params[i]->grad;
            m_adam[i]=beta1*m_adam[i]+(1-beta1)*g;
            v_adam[i]=beta2*v_adam[i]+(1-beta2)*g*g;
            double mh=m_adam[i]/(1-std::pow(beta1,step+1));
            double vh=v_adam[i]/(1-std::pow(beta2,step+1));
            params[i]->data-=lr_t*mh/(std::sqrt(vh)+eps);
            params[i]->grad=0.0;
        }
        std::cout<<"\rStep "<<(step+1)<<" / "<<num_steps<<"  loss = "<<loss->data<<std::flush;
    }
    std::cout<<"\n---------Inference---------\n";

    double temperature=0.5;
    for(int si=0;si<20;++si){
        std::vector<std::vector<Vec>> keys(N_LAYER),vals(N_LAYER);
        int token_id=BOS_ID; std::string sample;
        for(int pos=0;pos<SLIDING_WIN;++pos){
            Vec logits=gpt_forward(token_id,pos,keys,vals);
            Vec scaled; for(auto& l:logits) scaled.push_back(vscale(l,1.0/temperature));
            Vec probs=softmax(scaled);
            std::vector<double> w; for(auto& p:probs) w.push_back(p->data);
            std::discrete_distribution<int> dist(w.begin(),w.end());
            token_id=dist(rng);
            if(token_id==BOS_ID) break;
            sample+=uchars[token_id];
        }
        std::cout<<"Sample "<<(si+1)<<": "<<sample<<"\n";
    }
    return 0;
}
