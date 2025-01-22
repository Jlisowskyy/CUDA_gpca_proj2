#ifndef GLOBAL_CONF_HPP
#define GLOBAL_CONF_HPP

struct ConfigStruct {
    bool WriteDotFiles = false;
    bool Verbose = false;
};

extern ConfigStruct GlobalConfig;

#endif //GLOBAL_CONF_HPP
