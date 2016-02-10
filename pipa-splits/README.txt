
This package provides additional splits on the People In Photo Albums (PIPA) dataset. The new splits are designed in such a way that various person recognition scenarios are captured. 

___

Brief description of each split:

Original
Train and test a person's appearance on the same days (proposed in the original PIPA).

Album
Train a person's appearance in one album, test on the other album.

Time
Train a person's appearance in the earliest time range, test on the latest time range (using photo-taken-date metadata).

Day
Train and test a person's appearance on different days (manually separated).

___

The split files are formatted as in the original PIPA annotation with one extra information, split:
<photoset_id> <photo_id> <xmin> <ymin> <width> <height> <identity_id> <subset_id>  <SPLIT(0/1)>


Following the PIPA protocol, your identity classifier must be trained on split 0 instances and be tested on split 1 instances. Then, do the same with the split 0 and 1 swapped. For evaluation, take the average of the these performances.

___

For further questions, please contact: 

Seong Joon Oh <joon@mpi-inf.mpg.de>

