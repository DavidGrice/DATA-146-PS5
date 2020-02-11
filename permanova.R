#If you have not done so already either visually in RStudio or from R command line
#This only has to be done ONE TIME
#install vegan package - hit n if it asks to compile, y to install dependencies
#install.packages("vegan")
#load vegan package into session - must be done EVERY TIME
#library(vegan)
#Read in data with RStudio or from command line - you will need to make sure
#you are in the correct directory if working from the command line
#South_With_Category_Ints <- read.csv("South_With_Category_Ints.csv")
#You will need to supply the appropriate categories in the line below, these will be
#selected from 'A_Lot', 'Some', 'Not_Much', 'Not_At_All'
data1 <-subset(South_With_Category_Ints, category == 'A Lot' | category == 'Not_At_All')
int_data = data1[1:25] #do not change
cats = factor(unlist(data1[26])) #do not change
#in the line below you will need to pick the appropriate method for distance - see
#documentation here: http://cc.oulu.fi/~jarioksa/softhelp/vegan/html/vegdist.html
#change method = to the parameter you pick

adonis(int_data~cats, method='bray', permutations = 200)

data2 <-subset(South_With_Category_Ints, category == 'Some' | category == 'Not_At_All')
int_data2 = data2[1:25] #do not change
cats2 = factor(unlist(data2[26])) #do not change

adonis(int_data2~cats2, method='bray', permutations = 200)

data3 <-subset(South_With_Category_Ints, category == 'A Lot' | category == 'Some')
int_data3 = data3[1:25] #do not change
cats3 = factor(unlist(data3[26])) #do not change

adonis(int_data3~cats3, method='bray', permutations = 200)
