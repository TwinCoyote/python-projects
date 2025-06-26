import pkg_resources
print([pkg.key for pkg in pkg_resources.working_set])
