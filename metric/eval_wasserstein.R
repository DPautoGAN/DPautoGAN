library(transport)
score <- replicate(100,0)
for (i in c(0:99)){
  mydata <- read.csv(paste(i,'.csv', sep = ''), header = FALSE, sep = ',')
  M <- data.matrix(mydata)
  len <- dim(M)
  score[i+1] <- wasserstein(replicate(len[1], 1/len[1]), replicate(len[2], 1/len[2]), costm = M)
}
mean(score)