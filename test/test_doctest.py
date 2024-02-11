import doctest
import inspect
import importlib


def _get_module_by_string_lookup(module_str):
    return importlib.import_module(module_str)


def _get_submodules_tuples_from_module(module):
    return inspect.getmembers(languagemodels, inspect.ismodule)


module_str = "languagemodels"
languagemodels = _get_module_by_string_lookup(module_str)
modules = _get_submodules_tuples_from_module(languagemodels)

# Generate list with module and submodules to test
test_modules = [languagemodels]
for module_tup in modules:
    path = module_tup[1].__file__
    if module_str in path:
        submodule_name = f"{module_str}.{module_tup[0]}"
        test_modules.append(
            _get_module_by_string_lookup(submodule_name))
print(f"Modules to be tested: {test_modules}")

for module in test_modules:
    print(f"Testing module: {module}")
    doctest.testmod(module)
