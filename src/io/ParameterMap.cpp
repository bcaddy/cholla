#include "../io/ParameterMap.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

#include "../global/global.h"  // MAXLEN
#include "../io/io.h"          // chprintf
#include "../utils/error_handling.h"

[[noreturn]] void param_details::Report_TypeErr_(const std::string& param, const std::string& str,
                                                 const std::string& dtype, param_details::TypeErr type_convert_err)
{
  std::string r;
  using param_details::TypeErr;
  switch (type_convert_err) {
    case TypeErr::none:
      r = "";
      break;  // this shouldn't happen
    case TypeErr::generic:
      r = "invalid value";
      break;
    case TypeErr::boolean:
      r = R"(boolean values must be "true" or "false")";
      break;
    case TypeErr::out_of_range:
      r = "out of range";
      break;
  }
  CHOLLA_ERROR("error interpretting \"%s\", the value of the \"%s\" parameter, as a %s: %s", str.c_str(), param.c_str(),
               dtype.c_str(), r.c_str());
}

param_details::TypeErr param_details::try_bool_(const std::string& str, bool& val)
{
  if (str == "true") {
    val = true;
  } else if (str == "false") {
    val = false;
  } else {
    return param_details::TypeErr::boolean;
  }
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_int64_(const std::string& str, std::int64_t& val)
{
  char* ptr_end{};
  errno         = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  long long tmp = std::strtoll(str.data(), &ptr_end, 10);  // the last arg specifies base-10

  if (errno == ERANGE) {  // deal with errno first, so we don't accidentally overwrite it
    // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
    return param_details::TypeErr::out_of_range;
  } else if ((str.data() + str.size()) != ptr_end) {
    // when str.data() == ptr_end, then no conversion was performed.
    // when (str.data() + str.size()) != ptr_end, str could hold a float or look like "123abc"
    return param_details::TypeErr::generic;
#if (LLONG_MIN != INT64_MIN) || (LLONG_MAX != INT64_MAX)
  } else if ((tmp < INT64_MIN) and (tmp > INT64_MAX)) {
    return param_details::TypeErr::out_of_range;
#endif
  }
  val = std::int64_t(tmp);
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_double_(const std::string& str, double& val)
{
  char* ptr_end{};
  errno = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  val   = std::strtod(str.data(), &ptr_end);

  if (errno == ERANGE) {  // deal with errno first, so we don't accidentally overwrite it
    // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
    return param_details::TypeErr::out_of_range;
  } else if ((str.data() + str.size()) != ptr_end) {
    // when str.data() == ptr_end, then no conversion was performed.
    // when (str.data() + str.size()) != ptr_end, str could look like "123abc"
    return param_details::TypeErr::generic;
  }
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_string_(const std::string& str, std::string& val)
{
  // mostly just exists for consistency (every parameter can be considered a string)
  // note: we may want to consider removing surrounding quotation marks in the future
  val = str;  // we make a copy for the sake of consistency
  return param_details::TypeErr::none;
}

/*! \brief Gets rid of trailing and leading whitespace. */
static char* my_trim(char* s)
{
  /* Initialize start, end pointers */
  char *s1 = s, *s2 = &s[strlen(s) - 1];

  /* Trim and delimit right side */
  while ((std::isspace(*s2)) && (s2 >= s1)) {
    s2--;
  }
  *(s2 + 1) = '\0';

  /* Trim left side */
  while ((std::isspace(*s1)) && (s1 < s2)) {
    s1++;
  }

  /* Copy finished string */
  std::strcpy(s, s1);
  return s;
}

ParameterMap::ParameterMap(std::FILE* fp, int argc, char** argv)
{
  int buf;
  char *s, buff[256];

  CHOLLA_ASSERT(fp != nullptr, "ParameterMap was passed a nullptr rather than an actual file object");

  /* Read next line */
  while ((s = fgets(buff, sizeof buff, fp)) != NULL) {
    /* Skip blank lines and comments */
    if (buff[0] == '\n' || buff[0] == '#' || buff[0] == ';') {
      continue;
    }

    /* Parse name/value pair from line */
    char name[MAXLEN], value[MAXLEN];
    s = std::strtok(buff, "=");
    if (s == NULL) {
      continue;
    } else {
      std::strncpy(name, s, MAXLEN);
    }
    s = std::strtok(NULL, "=");
    if (s == NULL) {
      continue;
    } else {
      std::strncpy(value, s, MAXLEN);
    }
    my_trim(value);
    entries_[std::string(name)] = {std::string(value), false};
  }

  // Parse overriding args from command line
  for (int i = 0; i < argc; ++i) {
    char name[MAXLEN], value[MAXLEN];
    s = std::strtok(argv[i], "=");
    if (s == NULL) {
      continue;
    } else {
      std::strncpy(name, s, MAXLEN);
    }
    s = std::strtok(NULL, "=");
    if (s == NULL) {
      continue;
    } else {
      std::strncpy(value, s, MAXLEN);
    }
    entries_[std::string(name)] = {std::string(value), false};
    chprintf("Override with %s=%s\n", name, value);
  }
}

int ParameterMap::warn_unused_parameters(const std::set<std::string>& ignore_params, bool abort_on_warning,
                                         bool suppress_warning_msg) const
{
  int unused_params = 0;
  for (const auto& kv_pair : entries_) {
    const std::string& name                     = kv_pair.first;
    const ParameterMap::ParamEntry& param_entry = kv_pair.second;

    if ((not param_entry.accessed) and (ignore_params.find(name) == ignore_params.end())) {
      unused_params++;
      const std::string& value = param_entry.param_str;
      if (abort_on_warning) {
        CHOLLA_ERROR("%s/%s:  Unknown parameter/value pair!", name.c_str(), value.c_str());
      } else if (not suppress_warning_msg) {
        chprintf("WARNING: %s/%s:  Unknown parameter/value pair!\n", name.c_str(), value.c_str());
      }
    }
  }
  return unused_params;
}