# team-project
# Data-Driven Exploration of Survival Patterns on the Titanic

**Fall 2025 Data Science Project**  
**Johann Kuruvilla & Raghavendra Pavan Sunkara**

---

## Contributions

**Johann Kuruvilla**  
Johann Kuruvilla: Johann was primarily responsible for acquiring and preparing the Titanic Passenger and Crew dataset (B). Johann then led the data cleaning and preprocessing process (C). Following this, he conducted exploratory data analysis to examine patterns in survival outcomes, including relationships with gender, class, and fare, using statistical tests, and created visualizations to highlight these trends. He also interpreted the results of these analyses, explaining how differences in demographics and economic status affected survival odds. Finally, he helped compile and format the final report and tutorial, integrating all sections into a cohesive document that clearly communicates the methodology, analyses, and insights.

**Raghavendra Pavan Sunkara**  
Raghavendra Pavan Sunkara: Worked on ML Algorithm Design/Development (D), ML Algorithm Training and Test Data Analysis (E), and Visualization, Result Analysis, Conclusion (F) on this assignment. Pavan made the decision to utilize logistic regression for the ML model to see how each feature impacted odds of survival for passengers that had a specific feature. Additionally, Pavan trained the dataset and then put together a graph illustrating each feature’s survival odds (how a given feature affected passengers within the group’s chances of survival).

**Collaboration**  
Johann and Pavan worked together on various parts of the tutorial to ensure consistency and clarity in the final document.

---

## Introduction

  The sinking of the RMS Titanic in 1912 remains one of the most infamous maritime disasters in modern history. The records of who survived and who passed away in the tragedy provide an opportunity to investigate how social, economic, and demographic factors shaped each person’s chances of surviving the Titanic’s sinking. In this project, we used the Titanic Passenger and Crew dataset from Kaggle to explore whether characteristics such as gender, socioeconomic class, and ticket cost had any correlation with odds of survival.
  Our analysis centers around three guiding research questions:
  1. Did survival likelihood differ between men and women aboard the Titanic?
  2. Was passenger class associated with the probability of survival?
  3. Were passengers who paid higher fares more likely to survive?
  These questions matter because they connect the historical event to broader issues of inequality, resource allocation, and structural privilege. Disasters often magnify existing social hierarchies, and by examining patterns on the Titanic, we can better understand how demographic and economic factors make a difference in crisis survival.
  To address these questions, we constructed a complete data science pipeline. The dataset was acquired directly through KaggleHub for reproducibility, cleaned extensively to resolve missing or inconsistent values, and preprocessed to standardize key variables such as gender, class, and fare price. We then applied statistical methods tailored to the type of data and research question: a Chi-square test to compare survival rates between men and women, a one-way ANOVA to compare survival rates across first, second, and third class passengers, and a Mann-Whitney U test to examine differences in fare prices between those who survived and those who passed away..
  Together, these analyses allow us to evaluate the role that gender, socioeconomic status, and wealth had on a given passenger’s odds of surviving the Titanic’s sinking. The results show distinct patterns that reflected the social context of 1912.

---

## Data Curation

  Before conducting any statistical analysis, the Titanic dataset required extensive cleaning and preprocessing to ensure accuracy, consistency, and interpretability. The raw Passenger and Crew dataset, downloaded programmatically using KaggleHub, contained over two thousand entries with varying levels of completeness, a mixture of numeric and text-based fields, and inconsistent formatting. These issues rendered the existing dataset unusable, so data curation was required in order to standardize the dataset and deal with malformed data that could impede the data analysis process.
  The dataset was downloaded directly from Kaggle. After extraction, the main data file, PassengerCrew.csv, was loaded into a pandas DataFrame. After looking at the data set, we saw that essential demographic, socioeconomic, and occupational variables, such as class, gender, fare price, and each passenger’s fate were available within the dataset.
  To maintain consistency and prevent errors during analysis, we got rid of all of the whitespace in each column’s name. Duplicate records were removed to prevent potential group misinterpretation. Rows missing essential variables (such as Status, Crew/Passenger, Gender, and Fare Price) were also dropped, since these fields were required to separate data into the different groups that we wanted to look at. We also deleted the “Profile on Encyclopedia Titanica” column entirely because it contained links that did not matter for the purposes of our data analysis.

**Dataset citations:**  
- [Johann's Kaggle Dataset](https://www.kaggle.com/datasets/johannkay/titanic-passenger-and-crew-data)  
- [Pavan's Kaggle Dataset](https://www.kaggle.com/datasets/pavansunkara082/titanic-passengers-and-crew-data/data)

  The Fare Price column involved a complex data curation process, more so than any of the other columns. The dataset stored fares in many different British currency formats, often mixing pounds, shillings, and pence in text-based entries. To ensure that the inconsistent currency formats did not pose a problem in data analysis, we created a parsing function that removed the pound symbol and then extracted the value for the pounds, shillings, or pence quantity using regular expressions. We then converted all values into a consistent numeric format in decimal pounds, where 20 shillings/240 pence were both equal to 1 pound.
  Missing fare values were imputed using a hierarchical strategy designed to preserve the dataset’s underlying economic structure. First, any missing value was replaced with the median fare of the Class/Job group that a given passenger belonged to. If that information was unavailable, the median fare of only the passenger class that the passenger belonged to was used. If class-based medians were not sufficient, the median for the given job category was used instead, and any remaining gaps were finally resolved using the overall dataset median. This multi-stage approach made sure that each data point relied on the most contextually meaningful information possible while maintaining internal consistency in the fare distribution.
  Certain entries did not have values for the Class / Department column. Because passenger class was necessary for our ANOVA analysis and reflected socioeconomic status, it was important to reconstruct these values carefully. We coded an inference function that identified the most common class amongst passengers sharing the same job, tried to infer class based on the person’s job if the most common class for passengers sharing the same job was unavailable, and if that wasn’t available, then the nearest fare price by absolute difference was found, and the given entry was assigned the class that provided the closest match. This allowed us to preserve both occupational and economic relationships in the reconstructed class assignments.
  Gender labels were inconsistent (some of these labels could show as any of “M”, “man”, “Women”, “f”), so all entries were converted to lowercase and mapped to standardized categories (“male”, “female”). Only rows with valid gender and survival information were retained.
  The survival outcome was then converted into a binary variable, with 1 representing a survivor and 0 representing a victim. This binary encoding was required in order to successfully perform the statistical tests that were performed later on.
  Additional filtering was done, depending on the analysis that was being performed. For the ANOVA test, only passengers in 1st, 2nd, or 3rd class were included. For the Mann-Whitney U test, only rows with valid fare and survival values were preserved. These tailored subsets ensured that each test used the appropriate data and preserved the integrity of categorical groupings.

---

## Exploratory Data Analysis

  After curating and preparing the Titanic Passenger and Crew dataset, we conducted exploratory data analysis to better understand the structure of the data, identify key patterns, and justify our choice of statistical tests. Performing exploratory data analysis can show patterns between demographic, socioeconomic, and occupational variables and survival outcomes before applying any formal hypothesis testing.
  We began by comparing the amount of survivors to the amount of victims. A simple count plot or value counts summary showed that the majority of individuals aboard the Titanic did not survive. This imbalance is important for interpreting later analyses, particularly due to the fact that several statistical tests are sensitive to group size differences.
  Firstly, we explored survival rates of both men and women. The data showed that there was almost an equal number of men and women on the Titanic. When survival rates were blotted on a bar graph, it showed that the women of the Titanic had a much higher survival rate than the men on board the Titanic did. This pattern strongly suggested that there was a correlation between gender and survival odds, motivating our Chi-squared test later on.
  We also examined the distribution of passengers across the three main classes: 1st, 2nd, and 3rd class. While all three classes were represented, 3rd class had the largest number of individuals, followed by 1st and finally 2nd class. This distribution is historically accurate and helps contextualize the socioeconomic hierarchy aboard the Titanic.
  Plotting survival rates for each class during EDA revealed a visible gradient. Survival was highest in 1st class, moderate in 2nd, and lowest in 3rd class. This clear trend justified the use of one-way ANOVA to statistically test differences between the class survival means.
  Because fare values reflect socioeconomic standing in tandem with passenger class, we explored the distribution of Fare Price using histograms and boxplots. The distribution was heavily right-skewed, with most passengers paying relatively low fares and a small number of wealthy passengers paying extremely high prices. This non-normal distribution supported our choice of a non-parametric test (Mann-Whitney U) later when comparing fares between survivors and non-survivors.
  During EDA, plotting fare distributions by survival status showed that survivors tended to fall within higher fare ranges, although there was overlap between groups. This pattern helped reaffirm our hypothesis that higher economic status likely increased chance at survival.
  To understand the relationships between key variables, we also looked at pairwise visualizations such as survival vs. gender, survival vs. class, fare vs. class, and fare vs. survival. 
  A violin plot or boxplot comparing fares by class showed clear separation between the economic tiers, confirming that fare is a reliable numeric representation aligning with socioeconomic status.
  Additionally, pivot tables and grouped summary stats were used to compute average survival rates for each demographic group, allowing a quick comparison of patterns before applying formal tests.
  Exploratory data analysis also allowed us to verify that fare values were reasonable after conversion from pounds, shillings, and pence; no extreme missing-data patterns remained after imputation; and that the inferred class assignments behaved consistently (e.g., inferred 1st-class fares fell within typical 1st-class fare ranges). This step gave confidence that the dataset was ready for inferential statistics.

---

## Primary Analysis

  The machine learning technique that we chose to use was logistic regression. We decided to use logistic regression because we wanted to see how a passenger with a given feature would have their odds of survival improve or worsen. It assigned each feature (1st class, 2nd class, 3rd class, male passenger, female passenger) an odds ratio. An odds ratio of 1 indicated that a passenger with a given feature would not see their survival odds improve or worsen if they have that feature, with odds ratios greater than 1 seeing an improved survival rate from having that feature, and an odds ratio less than 1 seeing worsened survival chances due to having that feature. 

---

## Visualization

  The graph shows that female and 1st class passengers had an odds ratio of exactly 1, setting both categories as the baseline for survival likelihood. Being a second class passenger came with an odds ratio of just above 0.4, meaning that one’s odds of survival worsened quite a bit in second class. Being a third class passenger worsened survival odds a good amount than being in second class did, with an odds ratio just above 0.2. Being a male passenger was the worst thing for one’s survival odds, carrying an odds ratio of less than 0.1.

---

## Insights and Conclusions

  A reader who knows nothing about how certain features like passenger class and gender influenced survival rate would be able to learn more about how these features affected survival rate after reading it. It shows the reader that survival rates were uneven amongst gender as well as class and fare price, giving the reader the conclusion that females had a much higher survival rate than males did, and that the higher class passenger groups also had higher survivor rates than the lower class passenger groups. In short, an uninformed reader would learn that females had higher survival odds than males did, and would also learn that the higher a passenger’s class was, the greater their odds of survival
	A reader that already knew about the topic would not know it if they only knew the most basic facts of the disaster, and did not really learn much about how survival differed among various different passenger demographics.
