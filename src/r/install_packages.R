# Installs the R packages used by titanic_model.R.
#
# The rocker/r-ver:4.4.2 base image points CRAN at a dated snapshot
# (https://p3m.dev/cran/2025-02-27), so install.packages() always resolves
# to the same versions. The check below fails the Docker build if that changes.

required <- c(
  dplyr = "1.1.4",
  readr = "2.1.5"
)

cat("CRAN repository:", getOption("repos")[["CRAN"]], "\n")

install.packages(names(required), Ncpus = max(1L, parallel::detectCores()))

for (pkg in names(required)) {
  installed <- as.character(packageVersion(pkg))
  if (installed != required[[pkg]]) {
    stop(sprintf("%s %s was installed but %s is required", pkg, installed, required[[pkg]]))
  }
  cat(sprintf("Installed %s %s\n", pkg, installed))
}
