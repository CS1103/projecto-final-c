
//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include <numeric>
using namespace utec::neural_network;

namespace utec::neural_network {

    template<typename T>
    class Dense : public ILayer<T> {
        Tensor<T, 2> w, dw;
        Tensor<T, 2> b, db;
        Tensor<T, 2> last_x;

        using WeightInitializer = std::function<void(Tensor<T,2>&)>;
        using BiasInitializer = std::function<void(Tensor<T,2>&)>;

    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_feats, size_t out_feats, InitWFun init_w_fun, InitBFun init_b_fun ) {
            w = Tensor<T, 2>(std::array{in_feats, out_feats});
            dw = Tensor<T, 2>(std::array{in_feats, out_feats});
            b = Tensor<T, 2>(std::array<size_t,2>{1, out_feats});
            db = Tensor<T, 2>(std::array<size_t,2>{1, out_feats});

            init_w_fun(w);
            init_b_fun(b);
        }
        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_x = x;
            Tensor<T,2> output(std::array<size_t,2>{x.shape()[0], w.shape()[1]});

            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < w.shape()[1]; ++j) {
                    output(i,j) = b(0, j);
                    for (size_t k = 0; k < w.shape()[0]; ++k)
                        output(i,j) += x(i,k) * w(k,j);
                }
            }
            return output;
        }
        Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
            const auto& x = last_x;
            Tensor<T, 2> grad_input(std::array<size_t,2>{x.shape()[0], w.shape()[0]});

            dw.fill(0);
            db.fill(0);
            
            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t k = 0; k < w.shape()[0]; ++k) {
                    grad_input(i,k) = 0;
                    for (size_t j = 0; j < w.shape()[1]; ++j)
                        grad_input(i,k) += grad(i,j) * w(k,j);
                }
            }

            for (size_t k = 0; k < w.shape()[0]; ++k) {
                for (size_t j = 0; j < w.shape()[1]; ++j) {
                    dw(k, j) = 0;
                    for (size_t i = 0; i < x.shape()[0]; ++i)
                        dw(k,j) += x(i,k) * grad(i,j);
                }
            }

            for (size_t j = 0; j < w.shape()[1]; ++j) {
                db(0, j) = 0;
                for (size_t i = 0; i < x.shape()[0]; ++i)
                    db(0, j) += grad(i,j);
            }

            return grad_input;
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(w, dw);
            optimizer.update(b, db);
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H