library(ggplot2)
library(stringr)
library(plyr)
library(dplyr)
library(lubridate)
library(reshape2)
library(scales)
library(ggthemes)
library(Metrics)

setwd("/home/georgios/dynamic/output/video/mobilenetv2_pretrained")

data <- read.csv("test_predictions.csv", header = FALSE)
str(data)

dataNoAugmentation <- data[data$V2 == 0,]
str(dataNoAugmentation)

dataGlobalAugmentation <- data %>% group_by(V1) %>% summarize(meanPrediction = mean(V3), sdPred = sd(V3))
str(dataGlobalAugmentation)

sizeData <- read.csv("size.csv")
sizeData <- sizeData[sizeData$ComputerSmall == 1,]
str(sizeData)

sizeRelevantFrames <- sizeData[c(1,2)]
sizeRelevantFrames$Frame <- sizeRelevantFrames$Frame - 32
sizeRelevantFrames[sizeRelevantFrames$Frame < 0,]$Frame <- 0

beatByBeat <- merge(sizeRelevantFrames, data, by.x = c("Filename", "Frame"), by.y = c("V1", "V2"))
beatByBeat <- beatByBeat %>% group_by(Filename) %>% summarize(meanPrediction = mean(V3), sdPred = sd(V3))
str(beatByBeat)

fileLocation <- "/home/georgios/dynamic/a4c-video-dir/"
ActualNumbers <- read.csv(paste0(fileLocation, "FileList.csv", sep = ""))
ActualNumbers <- ActualNumbers[c(1,2)]
str(ActualNumbers)


dataNoAugmentation <- merge(dataNoAugmentation, ActualNumbers, by.x = "V1", by.y = "FileName", all.x = TRUE)
dataNoAugmentation$AbsErr <- abs(dataNoAugmentation$V3 - dataNoAugmentation$EF)
str(dataNoAugmentation)

beatByBeatAnalysis <- merge(sizeRelevantFrames, data, by.x = c("Filename", "Frame"), by.y = c("V1", "V2"))
str(beatByBeatAnalysis)


MAEdata <- data.frame(counter = 1:500)
MAEdata$sample <- -9999
MAEdata$error <- -9999

str(MAEdata)

for (i in 1:500){
  
  
  samplingBeat <-  sample_n(beatByBeatAnalysis %>% group_by(Filename), 1 + floor((i-1)/100), replace = TRUE) %>% group_by(Filename) %>% dplyr::summarize(meanPred = mean(V3))
  samplingBeat <- merge(samplingBeat, ActualNumbers, by.x = "Filename", by.y = "FileName", all.x = TRUE)
  samplingBeat$error <- abs(samplingBeat$meanPred - samplingBeat$EF)
  
  MAEdata$sample[i] <-  1 + floor((i-1)/100)
  MAEdata$error[i] <- mean(samplingBeat$error )
  
  
}

str(MAEdata)

beatBoxPlot <- ggplot(data = MAEdata) + geom_boxplot(aes(x = sample, y = error, group = sample), outlier.shape = NA
) + theme_classic() + theme(legend.position = "none", axis.text.y = element_text( size=7)) + xlab("Number of Sampled Beats") + ylab("Mean Absolute Error") + scale_fill_brewer(palette = "Set1", direction = -1) 

beatBoxPlot
