# ────────────────────────────────────────────────────────────────────────────────
# R script: Draw boxplots for self-fixing rate by SATD type for Java projects
# [Modified: Adjusted left label spacing and fixed '...' syntax error]
# ────────────────────────────────────────────────────────────────────────────────

# Clean environment
rm(list = ls())

# Install and load required packages
packages <- c("jsonlite", "data.table", "ggplot2", "scales")
installed <- installed.packages()[, "Package"]
to_install <- setdiff(packages, installed)
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(jsonlite)
library(data.table)
library(ggplot2)
library(scales) # For formatting Y-axis labels

# ——————————————————————————————————————————————————————————
# 1. Read raw JSON data (please modify according to actual path)
MERGED_DATA <- "R:/Code/Self-Fixed_SATD/Dataset/data/merged_data_40501_updated.json"
if (!file.exists(MERGED_DATA)) stop("File does not exist: ", MERGED_DATA)
raw <- fromJSON(MERGED_DATA, flatten = TRUE)
setDT(raw)

# ——————————————————————————————————————————————————————————
#
PROJECT_ID_COLUMN <- "project_name" # <--- [!!] Modify your project field name here

if (!PROJECT_ID_COLUMN %in% names(raw)) {
  stop(
    "Error: Field '", PROJECT_ID_COLUMN, "' not found in JSON data.\n",
    "Please modify line 42 of the script to the correct project identifier field name."
  )
}

# 2.1 Extract required data
dt <- raw[, .(
  project = get(PROJECT_ID_COLUMN),
  satd_type,
  is_self_fixed = as.integer(is_self_fixed)
)]

# 2.2 Calculate self-repair rate by project and SATD type
#     - .N is the total number of instances for this (project, type) combination
#     - sum(is_self_fixed) is the number of self-fixed instances in this combination
project_rates <- dt[, .(
  self_fix_rate = sum(is_self_fixed) / .N,
  total_issues_in_project = .N
), by = .(project, satd_type)]

# Filter out combinations with no instances
project_rates <- project_rates[total_issues_in_project > 0]

# ——————————————————————————————————————————————————————————
# 3. Calculate statistics for display below the chart

# 3.1 Total number of instances (from original dt)
total_counts <- dt[, .(total_issues = .N), by = satd_type]

# 3.2 Mean and median of project repair rates (from project_rates)
summary_stats <- project_rates[, .(
  average_value = mean(self_fix_rate),
  median_value = median(self_fix_rate)
), by = satd_type]

# 3.3 Merge
plot_summary <- merge(summary_stats, total_counts, by = "satd_type")

# 3.4 Convert to long format (Tidy Format) for ggplot plotting
summary_labels <- melt(
  plot_summary,
  id.vars = "satd_type",
  measure.vars = c("average_value", "median_value", "total_issues"),
  variable.name = "stat_type"
)

# 3.5 Add row titles
row_labels <- data.table(
  stat_type = c("average_value", "median_value", "total_issues"),
  label_text = c("Average value ", "Median value ", "Number of SATD "),
  
  # Originally 0.4, reduce this value to move labels left, away from Y-axis
  x_pos = 0.35 
)
summary_labels <- merge(summary_labels, row_labels, by = "stat_type")

# 3.6 Format values
summary_labels[stat_type == "average_value", value_text := sprintf("%.3f", value)]
summary_labels[stat_type == "median_value",  value_text := sprintf("%.3f", value)]
summary_labels[stat_type == "total_issues",  value_text := formatC(value, format = "d", big.mark = ",")]

# 3.7 Set Y-axis position (below 0 axis)
y_positions <- c("average_value" = -0.15, "median_value" = -0.25, "total_issues" = -0.35)
summary_labels[, y_pos := y_positions[stat_type]]

# 3.8 Handle row titles separately
row_labels[, y_pos := y_positions[stat_type]]

# ——————————————————————————————————————————————————————————
# 4. Draw boxplots (using ggplot2)

# Determine Y-axis extension range to leave space for text below
plot_margin_bottom <- -0.4

p <- ggplot() +
  
  # 4.1 Draw boxplots
  #     Use project_rates, Y-axis is self_fix_rate
  geom_boxplot(
    data = project_rates,
    aes(x = satd_type, y = self_fix_rate, fill = satd_type),
    color = "black",
    width = 0.6,
    alpha = 0.7, # Add transparency to make colors softer
    outlier.shape = NA # Hide outliers
  ) +
  scale_fill_brewer(palette = "Set2") + # Use softer color scheme
  
  # 4.2 Draw statistical values at the bottom
  geom_text(
    data = summary_labels,
    aes(x = satd_type, y = y_pos, label = value_text),
    size = 3.5,
    vjust = 0.5,
    color = "black"
  ) +
  
  # 4.3 Draw statistical row titles at the bottom
  geom_text(
    data = row_labels,
    aes(x = x_pos, y = y_pos, label = label_text),
    size = 3.5,
    vjust = 0.5,
    hjust = 1, # Right align
    color = "black"
  ) +
  
  # =================== [Modification point] ===================
  # 4.4 The annotate() function block for "Debt type"
  #     Along with redundant "..." has been completely removed to fix syntax error
  # ================================================
  
  # 4.5 Theme and beautification
  scale_y_continuous(
    name = "Self-repaying rate at project level",
    limits = c(plot_margin_bottom, 1.0), # Set Y-axis range
    breaks = seq(0, 1, 0.25),
    labels = scales::number_format(accuracy = 0.01) # Format Y-axis
  ) +
  scale_x_discrete(
    name = NULL, # X-axis title has been replaced by geom_text
    # Format X-axis labels (e.g., "defect debt" -> "Defect")
    labels = function(x) tools::toTitleCase(gsub(" debt", "", x)) 
  ) +
  theme_classic() + # Use classic theme, more concise and modern
  theme(
    # Increase left margin to make room for row titles
    plot.margin = margin(t = 15, r = 15, b = 15, l = 50, unit = "pt"), 
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5), # Add light gray horizontal grid lines
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    axis.text.x = element_text(size = 11, color = "black",), # Use default font
    axis.text.y = element_text(size = 10, color = "black"), # Use default font
    axis.title.y = element_text(size = 12, margin = margin(r = 10)), # Use default font, move up
    axis.line = element_line(color = "black", linewidth = 0.5),
    legend.position = "none" # Remove legend (because there's only one language)
  ) +
  
  # 4.6 Allow drawing outside the plot area (key! Make the table at the bottom visible)
  coord_cartesian(clip = "off")

# 5. Save image
OUTPUT_FILE <- "satd_self_fixing_rate_boxplot.pdf"
ggsave(OUTPUT_FILE, p, width = 8, height = 6, dpi = 300, bg = "white")

message(paste("✔ Boxplot generated:", OUTPUT_FILE))