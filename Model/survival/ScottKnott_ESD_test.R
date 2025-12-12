# ────────────────────────────────────────────────────────────────────────────────

# 0. 
rm(list = ls())

# 1. 
# (data.table, jsonlite, ScottKnottESD )
packages <- c("jsonlite", "data.table", "ScottKnottESD")
installed <- installed.packages()[, "Package"]
to_install <- setdiff(packages, installed)
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(jsonlite)
library(data.table)
library(ScottKnottESD) #  reshape2，

# 2.  JSON （）
MERGED_DATA <- "R:/Code/Self-Fixed_SATD/Dataset/data/merged_data_40501_updated.json"
if (!file.exists(MERGED_DATA)) stop("：", MERGED_DATA)
raw <- fromJSON(MERGED_DATA, flatten = TRUE)
DT  <- as.data.table(raw)[, .(satd_type, is_self_fixed, satd_survival_days)]

# ────────────────────────────────────────────────────────────────────────────────
# ： Scott–Knott ESD
# ────────────────────────────────────────────────────────────────────────────────
generate_sk_by_percentile <- function(sub_dt, out_file, alpha = 0.05) {
  
  # 1. 
  sub_dt[, satd_type := factor(satd_type)]
  sub_dt[, satd_survival_days := as.numeric(satd_survival_days)]
  
  # 2.   
  #      7(1), 30(1), 365(1)
  #     Table 5
  stats_summary <- sub_dt[, {
    total_n = .N
    .(
      mean_days = mean(satd_survival_days, na.rm = TRUE),
      median_days = as.numeric(median(satd_survival_days, na.rm = TRUE)),
      sample_count = total_n,
      pct_fixed_7d = sum(satd_survival_days <= 7) / total_n,
      pct_fixed_30d = sum(satd_survival_days <= 30) / total_n,
      pct_fixed_365d = sum(satd_survival_days <= 365) / total_n
    )
  }, by = satd_type]
  
  # 3.  SK-ESD 
  #    SK-ESD  (N x K) ，:
  #    - K  =  ( satd_type)
  #    - N  = 
  #    ，“” 3  (N=3)
  
  # 3a.  (Melt) 
  #     *** :  data.table::melt  ***
  sk_input_melted <- data.table::melt(stats_summary, 
                          id.vars = "satd_type", 
                          measure.vars = c("pct_fixed_7d", "pct_fixed_30d", "pct_fixed_365d"),
                          variable.name = "time_frame",
                          value.name = "percentage")
                          
  # 3b.  (Dcast)  (time_frame ~ satd_type)
  #     *** :  data.table::dcast  ***
  sk_input_wide <- data.table::dcast(sk_input_melted, 
                         formula = time_frame ~ satd_type, 
                         value.var = "percentage")
                         
  # 3c.  SK-ESD  data.frame
  #      sk_input_wide  data.table, 
  sk_input_wide[, time_frame := NULL] 
  mat <- as.data.frame(sk_input_wide)
  
  # 4.  Scott–Knott ESD
  #    (: N=3, ， )
  sk <- sk_esd(mat, alpha = alpha)
  
  # 5. 
  #     SK 
  sk_groups_dt <- data.table(
    satd_type = colnames(mat), 
    sk_group = as.integer(sk$groups)
  )
  
  # 6. 
  res <- merge(stats_summary, sk_groups_dt, by = "satd_type")
  
  # 7. 
  #    ，
  setorder(res, sk_group, -median_days)
  
  #    ，
  setcolorder(res, c("satd_type", 
                     "sk_group", 
                     "mean_days", 
                     "median_days", 
                     "sample_count", 
                     "pct_fixed_7d", 
                     "pct_fixed_30d", 
                     "pct_fixed_365d"))
  
  # 8.  JSON
  write_json(res,
             path       = out_file,
             pretty     = TRUE,
             auto_unbox = TRUE,
             force      = TRUE)
             
  message(paste("✔  (SK-ESD):", out_file))
}

# ────────────────────────────────────────────────────────────────────────────────
# 3.  (Self-Fixed)
# ────────────────────────────────────────────────────────────────────────────────
generate_sk_by_percentile(
  sub_dt   = DT[is_self_fixed == 1],
  # ，
  out_file = "ScottKnott_ESD_test_self_fixed_by_percent.json",
  alpha    = 0.05
)

# ────────────────────────────────────────────────────────────────────────────────
# 4.  (Non Self-Fixed)
# ────────────────────────────────────────────────────────────────────────────────
generate_sk_by_percentile(
  sub_dt   = DT[is_self_fixed == 0],
  # ，
  out_file = "ScottKnott_ESD_test_non_self_fixed_by_percent.json",
  alpha    = 0.05
)

message("✔✔✔ ！")