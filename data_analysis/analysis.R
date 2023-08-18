library(afex)
library(emmeans)
library(ggplot2)
library(psych)
library(PMCMR)
library(lmerTest)
library(performance)

set.seed(123)

data2 <- datafile_trees
filtered_data <- data2[data2$Condition == 1, c("maxReward", "it1Reward")]

data2$Condition<-as.factor(data2$Condition)
#data2$did_improve<-as.factor(data2$did_improve)
# data2$made_bad_trees<-as.factor(data2$made_bad_trees)
# data2$ID2 <- as.factor((data2$ID2))

data2$Domain = as.factor(data2$Domain)
# wilcox_result <- wilcox.test(filtered_data$it1Reward, filtered_data$it4Reward, paired = TRUE,exact = FALSE)
print(wilcox_result)
head(data2)
str(data2)

result <- friedman.test(did_improve ~ Domain | ID2, data = data2)
result
# first do normality checks

model <- lm(made_bad_trees ~Domain+A+E+N+O+C+DTFamiliarity+GamingFamiliarity+WeeklyHours + (1|ID2), data=data2)

# model <- lmer(maxReward ~ Domain * Condition + (1|ID2), data = data2)
# Load the necessary library for the test
# model <- aov_car(formula = maxReward ~ Condition * Domain + Error(ID2), data = data2)



anova_result <- anova(model)
print(anova_result)

studentized_residuals <- rstudent(model)
shapiro_stats <- shapiro.test(studentized_residuals)
#qqnorm(studentized_residuals); 
#qqline(studentized_residuals, col = 2)

p <- shapiro_stats$p.value
p
if (p < 0.05){
  print(cat('Not Normally Distributed -- BAD (' ,p, ')'))
} else {
  print(cat('Normally Distributed-- GOOD (',p,')'))
}

# Perform Shapiro-Wilk test for normality
shapiro_test <- by(data2$fluency, data2$Condition, shapiro.test)
print("Shapiro-Wilk Test for Normality:")
print(shapiro_test)

# Perform Levene's test for homoscedasticity
levene_test <- leveneTest(data2$fluency ~ data2$Condition)
print("Levene's Test for Homoscedasticity:")
print(levene_test)

levenes <- leveneTest(it4Reward ~ Condition, data = data2)
p <- levenes$'Pr(>F)'[1]
p
if (p < 0.05){
  print(cat('Heteroscedastic -- BAD (' ,p, ')'))
} else {
  print(cat('Homoscedastic -- GOOD (',p,')'))
}
# if any assumption fails
kruskal_test <- kruskal.test(did_improve ~ Condition, data=datafile)

# Print Friedman's test results
print(kruskal_test)
# Perform post-hoc pairwise comparisons using Dunn's test with Bonferroni adjustment
pairwise_dunn_test <- dunn.test::dunn.test(data2$it4Reward - data2$it1Reward, data2$Condition, method = "bh")

pairwise.wilcox.test(data2$it4Reward - data2$it1Reward,  data2$Condition,
                     p.adjust.method = "BH")
# Print pairwise comparison results
print("Pairwise Dunn's Test with Bonferroni Adjustment:")
print(pairwise_dunn_test)




anova_result <- anova(model)
print(anova_result)


# Perform post-hoc pairwise comparisons using Tukey's HSD test
posthoc <- glht(model, linfct = mcp(Condition = "Tukey"))
posthoc_results <- summary(posthoc)

# Print ANOVA table
print("ANOVA Table:")
print(anova_result)

# Print post-hoc pairwise comparison results
print("Post-hoc Pairwise Comparisons:")
print(posthoc_results)


