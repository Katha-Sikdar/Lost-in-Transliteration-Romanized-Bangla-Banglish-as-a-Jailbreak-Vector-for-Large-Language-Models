# Mixed-effects logistic regression for the main claim (Phase 5).
#
#   harmful ~ version * model + (1 | seed_id)
#
# Input: results/long.csv written by `python -m banglishjail.stats`.
# Run:   Rscript analysis/mixed_effects.R results/long.csv
# Needs: install.packages(c("lme4", "emmeans"))

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
})

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) > 0) args[1] else "results/long.csv"

d <- read.csv(path, stringsAsFactors = FALSE)
d <- subset(d, condition == "direct")
d$version <- relevel(factor(d$version), ref = "en")
d$model_name <- factor(d$model_name)
d$seed_id <- factor(d$seed_id)

fit <- glmer(harmful ~ version * model_name + (1 | seed_id),
             data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa"))
print(summary(fit))

# Odds ratios of each version vs English, averaged over models
emm <- emmeans(fit, ~ version)
print(contrast(emm, method = "trt.vs.ctrl", ref = "en", type = "response",
               adjust = "holm"))

# Per-model contrasts
emm_m <- emmeans(fit, ~ version | model_name)
print(contrast(emm_m, method = "trt.vs.ctrl", ref = "en", type = "response",
               adjust = "holm"))
