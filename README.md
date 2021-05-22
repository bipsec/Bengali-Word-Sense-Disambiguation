# Bengali-Word-Sense-Disambiguation
Due to recent technical and scientific advances in Bangla, Natural Language Processing plays a very important role in the field of Bengali Machine Translation, Information Extraction, Retrieval, and so on. Word Sense Disambiguation is one of the most emergent tasks in this field. This paper address the Disambiguation technique in Bengali Language and a well-annotated corpus has been developed for this purpose. In this paper, we have built a system design to detect the actual sense for each ambiguous word and the outcome of the assessment indicates that the word sense that we proposed is 79.6\%. However, when dealing with a large number of ambiguous words in a single language or technological domain, we can run into stumbling blocks when deploying existing WSD models.

## Corpus for Bengali Word Sense Disambiguation
Sample corpus! 

| Ambiguous Word | Meaning | Text |
| :---         |     :---:      | :---          |
| মাথা   | অঙ্গ বিশেষ     | মাথা মানবদেহে বিদ্যমান সবচেয়ে ভারি অংশ    |
| মাথা     | অভিভাবক       | গোপাল বাবু এই গাঁয়ের মাথা, তার কথা সবাই মেনে চলে  |
| মুখ    | মুখমণ্ডল       | সুন্দর মুখের প্রাথমিক শর্ত হল নিখুঁত, উজ্জ্বল ত্বক।  |
| মুখ    | সম্মান রাখা       | এই ছেলে একদিন বংশের মুখ রক্ষা করবে।  |
| পড়া    | পঠন       | গতকাল যে পড়া দিয়েছি, তা মুখস্ত কর নাই কেন? |
| পড়া    | স্মরণে আসা       | মা, তোমাকে খুব মনে পড়ার কারন জানি না |
| কড়া    | কঠোর       | কুলি-মজুরদের সঙ্গে কড়া আচরণ করা ঠিক না। |
| কড়া     | সতর্ক অর্থে       |  শিক্ষক পরীক্ষার হলে কড়া পাহারা থাকতেন। |
| গরম    | উত্তাপ      | গরম লাগছে, তাই হাইকোর্টে যাবেন না আইনজীবীরা |
| গরম    | মূল্য বৃদ্ধি        | ঈদ আসার বেশ আগে থেকেই গরম হয়ে আছে গরম মসলার বাজার।  |

## Building Vocab
Word embeddings are a way to describe terms in a semantically meaningful space as real-valued vectors. Pennington et al. (2014) presented Global Vectors for Word Representation (GloVe), a hybrid approach to embedding words. Word embeddings are able to collect fine-grained semantic and syntactic details about words and are educated unsupervised on vast quantities of data. Following that, these vectors can be used to initialize the input layer of a neural network or another NLP model. For this respect of work Bengali Glove Word Vectors have been used which have (39M(39055685) tokens, 0.18M(178152) vocab size.

## Training Details

To quantify how efficient and accurate our method is we also separated our dataset into different parts. The total size of our data set is 10,041 sentences based on more than thirty two ambiguous words having a minimum of ten
senses of each word. The dataset is split into training, cross-validation, and testing data. It is
split on a ratio of (70:20:10). From there, to assess the final result, we ran the different algorithms along with our written script. Finally, with the testing data, we evaluated our method.


## Evaluation Results
| Model  | Test Score | Validation Score
| ------------- | ------------- | ------------- |
| Naive Bayes  | 79.6%  | 75.97% |
| SVM  | 75.37%  | 74.3% |
| KNN  | 52.74%  | 51.93% |
| LSTM  | 78.67%  | 69.37% |
| Bi-LSTM + Glove | 78.73%  | 72.21% |


## Author
Biplab Kumar Sarkar, Afrar Jahin, Md Mahadi Hasan Nahid

## Acknowledgements
- Thanks to [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing the free TPU credits - thank you!
- Thank to all the people around, who always helping us to build something for Bengali.
