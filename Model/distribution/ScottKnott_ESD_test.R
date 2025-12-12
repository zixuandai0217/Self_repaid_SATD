# ────────────────────────────────────────────────────────────────────────────────
# R ： self-fixed  non_self-fixed SATD  Scott–Knott ESD 
# ────────────────────────────────────────────────────────────────────────────────

# 
rm(list = ls())

# 
packages <- c("jsonlite", "data.table", "ScottKnottESD")
installed <- installed.packages()[, "Package"]
to_install <- setdiff(packages, installed)
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)
library(jsonlite)
library(data.table)
library(ScottKnottESD)

# ——————————————————————————————————————————————————————————
# 1.  JSON （）
MERGED_DATA <- "R:/Code/Self-Fixed_SATD/Dataset/data/merged_data_40501_updated.json"
if (!file.exists(MERGED_DATA)) stop("：", MERGED_DATA)
raw <- fromJSON(MERGED_DATA, flatten = TRUE)

# ——————————————————————————————————————————————————————————
# 2.  data.table：satd_type + is_self_fixed
dt <- as.data.table(raw)[, .(satd_type, is_self_fixed)]
dt[, satd_type       := factor(satd_type)]
dt[, is_self_fixed   := as.integer(is_self_fixed)]

# 3. 
dt[, idx := seq_len(.N), by = satd_type]
wide_dt <- dcast(dt, idx ~ satd_type, value.var = "is_self_fixed")
setDT(wide_dt)
wide_mat <- as.data.frame(wide_dt[, -1, with = FALSE])  # ： satd_type， 1/0

# ——————————————————————————————————————————————————————————
# 4.  & 
self_fix_rate      <- colMeans(wide_mat,       na.rm = TRUE)
non_self_fix_rate  <- colMeans(1 - wide_mat,   na.rm = TRUE)

# 5. （self vs non-self）
counts <- dt[, .(
  self_fixed_count     = sum(is_self_fixed == 1, na.rm = TRUE),
  non_self_fixed_count = sum(is_self_fixed == 0, na.rm = TRUE)
), by = satd_type]

# ——————————————————————————————————————————————————————————
# 6. Scott–Knott ESD 
# 6.1  self-fixed (wide_mat)
sk_self <- sk_esd(wide_mat, alpha = 0.05)
groups_self <- sk_self$groups

# 6.2  non-self-fixed (1 - wide_mat)
wide_mat_non <- 1 - wide_mat
sk_non  <- sk_esd(wide_mat_non, alpha = 0.05)
groups_non <- sk_non$groups

# ——————————————————————————————————————————————————————————
# 7.  summary data.table  JSON

# 7.1 self-fixed
summary_self <- data.table(
  satd_type        = names(self_fix_rate),
  fix_rate         = as.numeric(self_fix_rate),
  sk_group         = as.integer(groups_self)
)
summary_self <- merge(summary_self, counts, by = "satd_type", all.x = TRUE)
write_json(
  summary_self,
  path       = "ScottKnott_ESD_test_self_fixed.json",
  pretty     = TRUE,
  auto_unbox = TRUE,
  force      = TRUE
)

# 7.2 non-self-fixed
summary_non <- data.table(
  satd_type        = names(non_self_fix_rate),
  fix_rate         = as.numeric(non_self_fix_rate),
  sk_group         = as.integer(groups_non)
)
summary_non <- merge(summary_non, counts, by = "satd_type", all.x = TRUE)
write_json(
  summary_non,
  path       = "ScottKnott_ESD_test_non_self_fixed.json",
  pretty     = TRUE,
  auto_unbox = TRUE,
  force      = TRUE
)

message("✔ ：ScottKnott_ESD_test_self_fixed.json  ScottKnott_ESD_test_non_self_fixed.json")
