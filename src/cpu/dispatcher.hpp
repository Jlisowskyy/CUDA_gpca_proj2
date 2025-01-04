#ifndef DISPATCHER_HPP
#define DISPATCHER_HPP

#include <arg_parser.hpp>

class Dispatcher {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    explicit Dispatcher(const ArgPack &arg_pack);

    ~Dispatcher() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    void Dispatch() const;

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    // ------------------------------
    // class fields
    // ------------------------------

    ArgPack arg_pack_;
};

#endif //DISPATCHER_HPP
