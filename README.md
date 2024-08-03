# worthnow
ai and ml for now
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load dataset
data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data.raw_ratings, reader)
trainset, testset = train_test_split(dataset, test_size=0.2)

# Train model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)
accuracy.rmse(predictions)
