#ifndef DATA_PARSER_HPP
#define DATA_PARSER_HPP

/* internal includes */
#include <data.hpp>

class DataParser {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    DataParser() = default;

    ~DataParser() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    [[nodiscard]] BinSequencePack ParseData(const char *path) const;

    void SaveData(const BinSequencePack &data, const char *path) const;

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    // ------------------------------
    // class fields
    // ------------------------------
};

#endif //DATA_PARSER_HPP
