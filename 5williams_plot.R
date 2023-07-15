setwd("I:/1caco2_reg")

library(openxlsx)
train<-read.csv("./PHD2_result/SVM_PHD2_RF113_train_error.csv")
test<-read.csv("./PHD2_result/SVM_PHD2_RF113_test_error.csv")


res1<-train[,1]-train[,2]
res2<-test[,1]-test[,2]


#ѵ����
williams<-data.frame(res1)
williams<-cbind(williams,diag(H1))
name1=rep("Training set",nrow(traindata))
williams<-cbind(williams,name1)

#���Լ�
test.williams<-data.frame(res2)
test.williams<-cbind(test.williams,diag(H2))
name2=rep("Test set",nrow(testdata))
test.williams=cbind(test.williams,name2)


#��ѵ�����Ͳ��Լ��ŵ�һ��
colnames(williams)[1]="Predict_Error"
colnames(williams)[2]="Leverage_Value"
colnames(williams)[3]="Data"
colnames(test.williams)[1]="Predict_Error"
colnames(test.williams)[2]="Leverage_Value"
colnames(test.williams)[3]="Data"
williams_all=merge(williams,test.williams,all=TRUE)

write.csv(williams_all,"./PHD2_result/williams_dot.csv")
print(williams_all)

#��׼���в�
sd_res=sd(williams_all[,1])
library(ggplot2)
# ggplot(data=williams_all,aes(x=Leverage_Value,y=Predict_Error,shape=Data,colour=Data))+

ggplot(williams_all, aes(x = Leverage_Value, y = Predict_Error,colour=Data)) +
  geom_point(alpha =0.5) +
  # scale_fill_distiller(type = "seq",palette = 8)
  
  
  geom_point(size=3)+geom_hline(aes(yintercept=-3*sd_res),colour="#CCCCCC",size=2)+
  geom_hline(aes(yintercept=3*sd_res),colour="#CCCCCC",size=2)+
  geom_vline(aes(xintercept=h),colour="#CCCCCC",size=2)+
  
  theme(axis.title= element_text(size=20, color="black", face="bold", vjust=0.5, hjust=0.5))+
  theme(axis.text = element_text(size=15,color="black"))+
  
  theme(legend.text = element_text(size = 15))+
  theme(legend.title = element_text(size = 15, face="bold"))+
  theme(legend.position=c(0,1),legend.justification=c(0,1))+
  theme(legend.background=element_rect(fill='grey92'))+
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))+
  labs(x = "Prediction Error", y = "Leverage Value")+
  ylim(-3,3)+
  xlim(0,1)+ 
  coord_flip() 

ggsave(file="./PHD2_result/williams plot.png",width = 10, height = 8, units = "cm")

