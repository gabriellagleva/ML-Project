library(tidyverse)

combined_data <- merge(CMMD_clinicaldata_revision, metadata, by.x="ID1", by.y="Subject ID")


FINALCLEANCMMD <- select(combined_data, c("ID1", "abnormality", "classification", "File Location"))

