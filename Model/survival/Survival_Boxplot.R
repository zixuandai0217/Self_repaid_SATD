# ──────────────────────────────────────────────────────────────────
# R Script: Generate high-quality boxplots (Boxplot) for SATD survival time
# Use ggplot2 package and adopt logarithmic coordinate axis to optimize display
# ──────────────────────────────────────────────────────────────────

# 0. Clean environment
rm(list = ls())

# 1. Install and load required packages
# data.table and jsonlite are from your previous script, added ggplot2 for plotting
packages <- c("jsonlite", "data.table", "ggplot2")
installed <- installed.packages()[, "Package"]
to_install <- setdiff(packages, installed)
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(jsonlite)
library(data.table)
library(ggplot2)

# 2. Read raw JSON data (please modify according to your actual path)
MERGED_DATA <- "R:/Code/Self-Fixed_SATD/Dataset/data/merged_data_40501_updated.json"
if (!file.exists(MERGED_DATA)) stop("File does not exist: ", MERGED_DATA)
raw <- fromJSON(MERGED_DATA, flatten = TRUE)
DT  <- as.data.table(raw)[, .(satd_type, is_self_fixed, satd_survival_days)]

# 3. Data preprocessing for plotting
# Convert is_self_fixed from 0/1 to meaningful factor labels
DT[, is_self_fixed := factor(
    is_self_fixed,
    levels = c(1, 0),
    labels = c("Self-Repaired", "Non Self-Repaired")
)]

# Convert satd_type to factor as well to ensure order when plotting
DT[, satd_type := factor(satd_type)]

# Ensure survival days are numeric
DT[, satd_survival_days := as.numeric(satd_survival_days)]


# 4. Draw boxplots (strongly recommended version: use logarithmic coordinate axis)
# ----------------------------------------------------
# Why use logarithmic coordinate axis?
# Because your data contains many extreme values (e.g., survival of thousands of days),
# ordinary coordinate axis will cause most boxes to be compressed at the bottom, making details unclear.
# We add 1 to the y-axis (satd_survival_days + 1) to prevent errors when taking logarithm when survival time is 0 days.
survival_boxplot <- ggplot(
  DT,
  # aes() defines the mapping relationship of the plot: x-axis is type, y-axis is survival days, color is also distinguished by type
  aes(x = satd_type, y = satd_survival_days + 1, fill = satd_type)
) +
  geom_boxplot(
    # outlier.shape = 21 can make outliers look better
    # outlier.size = 1.5 reduces the size of outliers to avoid overly cluttered images
    outlier.shape = 21, outlier.size = 1.5, outlier.alpha = 0.3
  ) +

  # *********** This is the most critical step ***********
  # Use logarithmic coordinate axis (base 10) and set scale labels
  scale_y_log10(
    breaks = c(1, 11, 101, 1001, 10001), # Define scale line positions (10^0, 10^1, ...)
    labels = c("0", "10", "100", "1000", "10000")   # Display scale lines as actual days
  ) +

  # *********** This is the second most critical step ***********
  # Use facet_wrap to split the plot into two panels by "is_self_fixed" variable
  # scales = "free_y" allows two panels to have different y-axis ranges, but here we keep them consistent for easy comparison
  facet_wrap(~ is_self_fixed, ncol = 2) +

  # --- Beautification and labels ---
  labs(
    # title = "Survival Days of Different SATD Types",
    # subtitle = "Comparison between Self-Fixed and Non Self-Fixed (Y-axis on Log Scale)",
    x = "SATD Type",
    y = "Survival Days (Log Scale)"
  ) +
  theme_bw(base_size = 14) + # Use a clean black and white theme
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), # Rotate x-axis labels 45 degrees to prevent overlap
    legend.position = "none" # Hide legend because color and x-axis information are redundant
  )

# 5. Display and save image
# Display image directly in RStudio
# print(survival_boxplot)

# Save as high-resolution PNG file, suitable for inserting into papers
ggsave(
  filename = "SATD_Survival_Boxplot_Log_Scale.pdf",
  plot = survival_boxplot,
  width = 12, # Width, in inches
  height = 7, # Height, in inches
  dpi = 300   # Resolution, 300dpi is printing standard
)

message("✔ Boxplot generated and saved as SATD_Survival_Boxplot_Log_Scale")