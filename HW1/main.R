library(readxl) #Run install.R if you don't have this library
# Step 1. Remove invaid entries. AKA has *,**,#
cat("#######STEP 1", "\n")
invalid <- c("**","*")#,"#")
nationalSal <- read_excel("NationalSalaries.xlsx")

invalid_rows <- apply(nationalSal, 1, function(row) {any(row %in% invalid)})
cleanNationalSal <- nationalSal[!invalid_rows, ]
cat("Rows before cleaning:", nrow(nationalSal), "\n")
cat("Rows after cleaning:", nrow(cleanNationalSal), "\n")
cat("Rows removed:", sum(invalid_rows), "\n")

#Step 2. Only columns in Salaries.xlsx
cat("#######STEP 2 - NEW CSV MADE", "\n")
sal <- read_excel("Salaries.xlsx")
keep <- c("ST", "STATE", "OCC_CODE", "OCC_TITLE", "GROUP", "TOT_EMP", "H_MEAN", "A_MEAN")
new_data <- cleanNationalSal[, keep]
write.csv(new_data, "Salaries_New.csv", row.names = FALSE)

#Step 3. Randomly select rows
cat("#######STEP 3", "\n")
set.seed(123)
random_indices <- sample(1:nrow(new_data), 1500)
sampled_data <- new_data[random_indices, ]
cat("Number of rows in random sample:", nrow(sampled_data), "\n")

#Step 4. Create data frame for low wage jobs
cat("#######STEP 4", "\n")
low_wage_jobs <- new_data[new_data$GROUP == "" & new_data$H_MEAN < 15, ]
cat("Number of rows meeting criteria:", nrow(low_wage_jobs), "\n")

#Step 5. Data frame for Indiana
cat("#######STEP 5", "\n")
indiana_jobs <- new_data[new_data$ST == "IN" & is.na(new_data$GROUP), ]
indiana_jobs$A_MEAN <- as.numeric(indiana_jobs$A_MEAN)
indiana_jobs <- indiana_jobs[!is.na(indiana_jobs$A_MEAN), ]
indiana_bins <- cut(indiana_jobs$A_MEAN, breaks = 10, dig.lab = 10)
salary_distribution <- table(indiana_bins)
cat("Number of Indiana jobs found in sample:", nrow(indiana_jobs), "\n")
print(salary_distribution)

#Step 6. Total State
cat("#######STEP 6", "\n")
new_data$TOT_EMP <- as.numeric(new_data$TOT_EMP)
state_totals_clean <- new_data[new_data$OCC_CODE == "00-0000", ]
state_employment <- state_totals_clean[order(-state_totals_clean$TOT_EMP), c("ST", "TOT_EMP")]
print(head(state_employment))

#Step 7. Average Salary
cat("#######STEP 7", "\n")
indiana_avg <- mean(indiana_jobs$A_MEAN, na.rm = TRUE)
cat("Calculated Indiana Average Yearly Salary:", indiana_avg, "\n")
cat("Comparison:  Average is", indiana_avg, "vs provided 42630/36410\n")

#Step 8. Chart
cat("#######STEP 8 - CHART MADE", "\n")
comp_math <- new_data[new_data$ST %in% c("IN", "CA", "NY") & grepl("^15-", new_data$OCC_CODE) & is.na(new_data$GROUP), ]
comp_math$A_MEAN <- as.numeric(comp_math$A_MEAN)
comp_math <- comp_math[!is.na(comp_math$A_MEAN), ]
state_comparison <- aggregate(A_MEAN ~ ST, data = comp_math, mean)
png("Salary_Comparison.png", width = 800, height = 600)
colors <- c("red", "blue", "green")
barplot(state_comparison$A_MEAN, 
        names.arg = state_comparison$ST, 
        col = colors,
        main = "Average Salary: Computer & Mathematical Occupations",
        xlab = "State", 
        ylab = "Mean Annual Salary ($)",
        ylim = c(0, max(state_comparison$A_MEAN) * 1.3))

legend("topright", 
       legend = state_comparison$ST, 
       fill = colors, 
       title = "States")
dev.off()