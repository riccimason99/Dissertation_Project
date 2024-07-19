library(tidyverse)
theme_set(theme_minimal())
library(ggplot2)





all_text <- read.csv('/Users/riccimason99/Downloads/Dissertation_2024/all_text_data_frame_clean.csv')
names(all_text)


# Plot the points
ggplot(all_text, aes(x=neg,
                     y=binary))+
  geom_jitter(height= 0.05,
              alpha = .1)

# Make the model
model <- glm(binary ~ neg, 
             data = all_text,
             family = "binomial")

summary(model)

# Plot data and regresion line 
ggplot(all_text, aes(x=neg,
                     y=binary))+
  geom_jitter(height= 0.05,
              alpha = .1) +
  geom_smooth(method = 'glm',
              method.args = list(family = 'binomial'))
              ,se = FALSE)


# Plot data and regresion line 
ggplot(all_text, aes(x=neg,
                     y=pred_prob))+
  geom_jitter(height= 0.05,
              alpha = .1) +
  geom_smooth(method = 'glm',
              method.args = list(family = 'binomial'))
,se = FALSE)
                    
                    
3.21 + 1 * 3.21



