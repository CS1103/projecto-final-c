#ifndef STATE_H
#define STATE_H

namespace utec::pong {

    struct State {
        float ball_x;
        float ball_y;
        float paddle_y;
        float dx;
        float dy;

        State(float bx = 0.0f, float by = 0.0f, float py = 0.0f, float dx_ = 0.0f, float dy_ = 0.0f)
            : ball_x(bx), ball_y(by), paddle_y(py), dx(dx_), dy(dy_) {}
    };

} // namespace utec::pong

#endif // STATE_H
