# A Machine Learning Framework for Understanding the Link between Parental Education, Socioeconomic Status, and Student Performance

## 📋 Title

**A Machine Learning Framework for Understanding the Link between Parental Education, Socioeconomic Status, and Student Performance**

*Presented at the 2nd International Conference on Artificial Intelligence & Sustainable Computing (AISC 2025), Kolkata, India*

## 🚀 Executive Summary

This study investigates how high school students' achievement is impacted by demographics, parental education, and socioeconomic factors through a comprehensive machine learning approach. Using a dataset of 1,000 students, we examine the effects of test preparation courses, lunch type (as SES proxy), parental education levels, gender, and race/ethnicity on math, reading, and writing scores.

**Key Findings:**
- Students with higher parental education consistently achieve better scores across all subjects
- Socioeconomic status (measured via lunch type) significantly impacts math and writing performance
- Test preparation completion provides measurable performance boosts (+7 points average)
- Gender differences show boys slightly leading in math while girls excel in reading and writing
- Linear Regression outperforms complex models with R² = 0.145, indicating scope for enhanced feature engineering

The results demonstrate substantial variation in academic achievement, with better scores in all areas being correlated with greater parental education levels. Students whose parents held bachelor's or master's degrees routinely outperformed classmates whose parents had less education.

## 🎯 Business Problem

**Challenge:** How can schools and policymakers identify the most impactful factors (parental education context, socioeconomic indicators, test preparation access, demographic variables) that influence student academic performance to optimize resource allocation and maximize learning outcomes?

**Impact:** Enable data-driven educational interventions that address systemic inequities in access to educational resources and improve academic success for all students through targeted, evidence-based programs.

**Research Questions:**
- What is the relationship between parental education levels and student academic performance across math, reading, and writing?
- How does socioeconomic status (proxied by lunch type) influence educational outcomes?
- Which machine learning models best predict student performance based on demographic and socioeconomic factors?
- What are the gender-based differences in academic performance across subjects?

## 🔬 Methodology

### 2.1 Dataset Description
The study utilized data from 1,000 high school students examining associations between student background characteristics and academic performance:

**Demographic Variables:**
- **Gender:** Binary variable (0 for male, 1 for female)
- **Race/Ethnicity:** Categorical groups based on standardized classifications (Groups A-E)

**Family Background:**
- **Parental Education:** Categorical variable indicating highest education level (High School → Master's Degree)

**Socioeconomic Factors:**
- **Lunch Type:** "Standard" or "Free/Reduced" serving as SES proxy
- **Test Preparation Course:** Binary indicator of completion (0 for none, 1 for completed)

**Academic Performance Variables:**
- **Math Score, Reading Score, Writing Score:** Range 0-100 on standardized tests
- **Average Score:** Overall academic performance measure

### 2.2 Data Preprocessing
- **Encoding:** Categorical variables converted to numerical formats using Label Encoding
- **Target Variable:** Average score calculated from math, reading, and writing scores
- **Split:** 70-30 train-test division for consistent model evaluation

### 2.3 Machine Learning Implementation
**Models Evaluated:**
- **Linear Regression:** Baseline model for performance prediction
- **Decision Tree Regressor:** Explore non-linear feature relationships
- **Random Forest Regressor:** Ensemble learning for improved prediction through multiple decision trees

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²) Score
- Cross-validation for robustness assessment

## 🛠️ Skills & Technologies

**Programming & Libraries:**
- Python for data analysis and machine learning
- Pandas for data manipulation and preprocessing
- Scikit-learn for model implementation and evaluation
- NumPy for numerical computations
- Matplotlib & Seaborn for data visualization

**Statistical Analysis:**
- Regression analysis and descriptive statistics
- Feature encoding and data preprocessing
- Cross-validation and model evaluation
- Performance metrics interpretation

**Machine Learning:**
- Supervised learning algorithm implementation
- Model comparison and selection
- Feature importance analysis
- Predictive modeling for educational outcomes

**Research Methodology:**
- Educational data analysis
- Socioeconomic impact assessment
- Statistical significance testing (p < 0.05)
- Policy-relevant insight generation

## 📊 Results & Business Recommendations

### Key Findings

**Table 1: Academic Performance by Parental Education Level**

| Parental Education Level | Math Score | Reading Score | Writing Score |
|-------------------------|------------|---------------|---------------|
| High School             | 60.1       | 63.5          | 61.4          |
| Associate's Degree      | 65.2       | 67.8          | 68.4          |
| Bachelor's Degree       | 72.6       | 74.3          | 75.1          |
| Master's Degree         | 78.3       | 80.2          | 82.0          |

**Achievement Gap Analysis:**
- **Educational Gap:** 18+ point difference between Master's and High School parental education
- **Gender Patterns:** Boys +2 points in math; Girls +3 points in reading/writing
- **SES Impact:** 5-7 point disadvantage for free/reduced lunch students
- **Test Preparation Effect:** +7 point boost across all subjects

**Table 2: Machine Learning Model Performance**

| Model             | MAE  | MSE   | R²    | Interpretation                    |
|-------------------|------|-------|-------|-----------------------------------|
| Linear Regression | 5.72 | 196.4 | 0.145 | Best performer, captures 14.5% variance |
| Decision Tree     | 6.84 | 248.1 | -0.032| Overfitting, poor generalization  |
| Random Forest     | 6.02 | 231.3 | 0.025 | Slight improvement over Decision Tree |

### Business Recommendations

**1. Target Socioeconomic Disparities:**
- Expand free/reduced lunch programs to address SES-based performance gaps
- Implement additional academic support specifically for low-SES students
- Potential impact: +5-7 point improvement in math and writing scores

**2. Scale Test Preparation Access:**
- Develop affordable test preparation programs for underserved communities
- Partner with community organizations for broader reach
- Demonstrated impact: +7 point boost across all academic subjects

**3. Enhance Family Engagement:**
- Create programs supporting parents with limited formal education
- Provide resources for enriched home learning environments
- Address the largest predictor: 18+ point parental education gap

**4. Implement Gender-Aware Interventions:**
- Strengthen reading and writing skill development programs for boys
- Maintain and enhance math confidence-building initiatives for girls
- Promote balanced achievement across all academic subjects

**5. Data Enhancement Strategy:**
- Collect additional variables: school quality metrics, student motivation indicators, teacher qualifications
- Current modest R² (0.145) indicates significant predictive potential remains untapped
- Invest in comprehensive data collection for improved intervention targeting

## 🚀 Next Steps

### Immediate Actions (0-6 months):
- Pilot expanded meal programs in high-need districts
- Launch community-based test preparation initiatives
- Implement family engagement workshops for parents with limited education
- Begin comprehensive data collection enhancement

### Medium-term Strategy (6-18 months):
- **Advanced Feature Engineering:** Integrate school quality, attendance patterns, extracurricular participation
- **Causal Analysis:** Implement randomized controlled trials to validate intervention effectiveness
- **Model Enhancement:** Explore neural networks, gradient boosting for improved predictive accuracy
- **Policy Integration:** Collaborate with education departments for evidence-based resource allocation

### Long-term Vision (1-3 years):
- **Real-time Analytics:** Deploy administrative dashboards for continuous monitoring and decision support
- **Longitudinal Studies:** Track intervention effectiveness over multiple academic years
- **Scalable Implementation:** Expand successful interventions across broader educational systems
- **Research Extension:** Include individual behavioral data, peer influence, and learning environment factors

## 👥 Authors & Affiliation

**Research Team:**
- **Swastika Manna¹** - swastikamanna03@gmail.com
- **Tiyas Biswas¹** - tiyasbi2003@gmail.com
- **Dipyendu Das¹** - dipyendudas5@gmail.com
- **Abhijit Pasari¹** - pasariabhijit425@gmail.com
- **Dr. Jayanta Pal¹** - jayanta.pal@gmail.com

¹ *Department of Computer Science and Engineering, Narula Institute of Technology, Kolkata, India*

## 🏆 Conference Publication

**Conference:** 2nd International Conference on Artificial Intelligence & Sustainable Computing (AISC 2025)  
**Location:** Kolkata, India  
**Dates:** July 24-26, 2025  
**Technical Sponsor:** IEEE Kolkata Section  
**Proceedings:** Submitted to Springer (camera-ready in progress)  

## 📚 Keywords

**Primary:** Academic Performance, Socioeconomic Status, Parental Education, Achievement Gap, Educational Equity, Machine Learning

**Technical:** Linear Regression, Decision Tree, Random Forest, Predictive Modeling, Educational Data Mining, Statistical Analysis

## 🔍 Research Impact

This study contributes to the growing field of educational data science by:

1. **Quantifying Achievement Gaps:** Providing precise measurements of educational disparities across demographic groups
2. **Validating Intervention Targets:** Confirming test preparation and family engagement as high-impact intervention points
3. **Informing Resource Allocation:** Enabling data-driven policy decisions for educational equity
4. **Advancing ML in Education:** Demonstrating both capabilities and limitations of current predictive approaches
5. **Establishing Baselines:** Creating benchmark performance metrics for future comparative studies

The research findings support targeted interventions addressing systemic inequities in educational access and outcomes, with clear policy implications for improving academic success across diverse student populations.

---

*This research was conducted as part of ongoing efforts to understand and address educational equity through data-driven approaches, presented at AISC 2025 with technical sponsorship from IEEE Kolkata Section and proceedings publication through Springer.*