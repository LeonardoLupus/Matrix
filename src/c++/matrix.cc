#include <matrix/matrix.h>


auto detail::warn(std::ostream &os, std::string_view message) -> std::ostream & {
  os << "Warning: " << message << std::endl;
  return os;
}
