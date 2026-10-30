# A device module's PTX inside a program, run as a script at build time (cmake -P, from optix.cmake's
# raytracing_embed_ptx) whenever the module's PTX was compiled again:
#   cmake -DBIN2C=<bin2c> -DNAME=<array> -DPTX=<module.ptx> -DCPP=<out.cpp> -DCOPY=<copy.ptx> -DSHA256=<copy.sha256>
#         -P embed.cmake
# writes CPP, a C++ file that defines, as src/gpu/optix/embedded.h declares them,
#   extern "C" const unsigned char NAME[];      the PTX's bytes, without a terminating zero
#   extern "C" const std::size_t NAME_size;     their number
#   extern "C" const char NAME_sha256[];        their SHA-256, 64 lowercase hex digits
# and copies the PTX to COPY with the same 64 digits and a newline in SHA256, which scripts/verify/irgate.ps1 compares
# with the copy and with the string in the programs. Every file is replaced only when its content changed, so a PTX
# that came out the same compiles and links nothing again.
foreach(variable IN ITEMS BIN2C NAME PTX CPP COPY SHA256)
    if(NOT ${variable})
        message(FATAL_ERROR "embed.cmake needs -D${variable}=...")
    endif()
endforeach()

# bin2c writes the array inside a braced `extern "C" { const unsigned char NAME[] = {...}; }`, where a const has
# internal linkage: the declarations before it give it the external linkage the runtime links against
execute_process(COMMAND "${BIN2C}" --const --type char --name "${NAME}" "${PTX}"
    OUTPUT_VARIABLE array RESULT_VARIABLE result ERROR_VARIABLE problem)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "bin2c could not write ${PTX} as an array: ${problem}")
endif()
file(SHA256 "${PTX}" sha256)

get_filename_component(source "${PTX}" NAME)
string(CONCAT text
    "// ${source} as an array (cmake/embed.cmake writes this file at every build of the module: don't edit it)\n"
    "#include <cstddef>\n\n"
    "extern \"C\" const unsigned char ${NAME}[];\n"
    "extern \"C\" const std::size_t ${NAME}_size;\n"
    "extern \"C\" const char ${NAME}_sha256[];\n\n"
    "${array}\n"
    "extern \"C\" const std::size_t ${NAME}_size = sizeof(${NAME});\n"
    "extern \"C\" const char ${NAME}_sha256[] = \"${sha256}\";\n")

# Written next to the targets, then copied over them only if different: the copy keeps the old file's time otherwise
file(WRITE "${CPP}.new" "${text}")
file(COPY_FILE "${CPP}.new" "${CPP}" ONLY_IF_DIFFERENT)
file(REMOVE "${CPP}.new")
file(COPY_FILE "${PTX}" "${COPY}" ONLY_IF_DIFFERENT)
file(WRITE "${SHA256}.new" "${sha256}\n")
file(COPY_FILE "${SHA256}.new" "${SHA256}" ONLY_IF_DIFFERENT)
file(REMOVE "${SHA256}.new")
