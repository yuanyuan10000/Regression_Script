caco2 <- read.csv("I:/1caco2_reg/PHD2_data/PHD2_value.csv")
attach(caco2)
library(ggplot2)


ggplot(caco2,aes(x = Label))+
  geom_density(aes(color = Class))+

  labs(x = "logIC50", y = "Density")+
  
  scale_color_brewer(palette = "Set1")+
  
  theme(legend.text = element_text(size = 10))+
  theme(legend.title = element_text(size = 10, face="bold"))+
  theme(legend.position=c(0,1),legend.justification=c(0,1))+
  theme(legend.background=element_rect(fill='grey92'))+
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))

ggsave(file="I:/1caco2_reg/PHD2_result/Density of Value_1.png",width = 10, height = 8, units = "cm")








