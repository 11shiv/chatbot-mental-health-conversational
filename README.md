# chatbot-mental-health-conversational

## Project Overview: Medical ChatBot Assistant

### Objective
The goal of the " Medical ChatBot Assistant " project is to create a simple chatbot that can assist users with medical queries and provide relevant information.

### Project Components and Flow

1. **Importing Libraries**: The project starts by importing essential libraries for data processing, machine learning, and natural language processing. These libraries include NumPy, pandas, PyTorch, and nltk (Natural Language Toolkit).

2. **Neural Network Model (NeuralNet)**: The core component of the project is the neural network model called `NeuralNet`. This model is defined using PyTorch's `nn.Module` class and consists of three linear layers with ReLU activation functions. The model takes a bag-of-words representation of input sentences and predicts the most suitable intent tag.

3. **Intent Data**: The project relies on a JSON file named `intents.json`. This file contains pre-defined intents, patterns (example sentences), and corresponding responses. Each intent has a tag that represents a specific medical topic or question.

4. **Text Preprocessing**: Text preprocessing functions are defined to tokenize input sentences (split them into words) and perform stemming (reduce words to their root form). These functions help in converting raw text into a format suitable for model input.

5. **Creating Training Data**: The project processes the intents data from `intents.json` and generates training data. Each input is converted into a bag-of-words representation, and the corresponding intent tag is used as the output label.

6. **Dataset and DataLoader**: To facilitate efficient training, a custom PyTorch `ChatDataset` class is created. This class handles the loading of training data and is utilized by the `DataLoader` to provide data batches for training.

7. **Model Training**: The neural network model is trained using the training data. The training loop iterates through the dataset for a specified number of epochs. During each iteration, the model computes predictions, calculates loss (using CrossEntropyLoss), performs backpropagation, and updates the model's parameters using an Adam optimizer.

8. **Model Saving**: After training, the model's state dictionary along with relevant information such as input size, hidden size, output size, words, and tags are saved to a file named `data.pth`. This saved file serves as the trained model's checkpoint.

9. **Model Loading**: During the chatbot's runtime, the saved model and associated information are loaded from `data.pth`. This allows the chatbot to use the trained model for predictions.

10. **User Interaction Loop**: The project enters a loop where the user can interact with the chatbot. The user inputs sentences related to medical topics or questions. The chatbot tokenizes the input, converts it into a bag-of-words representation, and uses the trained model to predict the intent tag. If the predicted intent's probability exceeds a certain threshold, the chatbot selects a response from the corresponding intent and presents it to the user.

11. **Exiting the Loop**: The user can exit the chatbot loop by typing "quit," which ends the interaction.

# Algorithm used :

1. **Bag-of-Words Approach**:
   - In this approach, each input sentence is represented as a vector that encodes the presence or absence of words from a predefined vocabulary.
   - Words are stemmed to their root form, and a vocabulary of stemmed words is created from the training data.
   - Tokenized words in input sentences are converted into their stemmed forms and matched against the vocabulary to create a bag-of-words representation.
   - Each element in the bag-of-words vector corresponds to whether a specific word from the vocabulary appears in the input sentence.

2. **Neural Network Model**:
   - The core algorithm involves training a neural network model to predict the intent of an input sentence based on its bag-of-words representation.
   - The neural network consists of three linear layers (fully connected layers) with ReLU activation functions in between.
   - The input layer's size corresponds to the size of the bag-of-words vector (the vocabulary size), and the output layer's size matches the number of unique intent tags.
   - The model is trained to minimize the cross-entropy loss between predicted intent probabilities and the true intent labels.

3. **Training and Prediction**:
   - During training, the neural network learns to identify patterns in the bag-of-words representations that are indicative of specific intents.
   - The training loop iterates over the dataset for a specified number of epochs, adjusting the model's parameters to minimize the loss.
   - During prediction, the bag-of-words representation of a user's input is passed through the trained model to predict the probabilities of different intents.
   - The predicted intent tag is the one with the highest probability.

4. **Threshold-based Response**:
   - A threshold probability is defined to filter out low-confidence predictions. If the predicted probability for a certain intent is below the threshold, the chatbot responds with a message indicating that it does not understand the query.
   - If the predicted probability is above the threshold, the chatbot selects a response associated with the predicted intent and presents it to the user.

5. **Data Persistence**:
   - The trained model's state dictionary, along with other project-related data (such as vocabulary, tags, and input/output sizes), is saved to a file named `data.pth` after training.
   - During chatbot runtime, the saved data is loaded to initialize the model and facilitate user interaction.

The algorithm's combination of a bag-of-words representation and a simple neural network demonstrates the project's basic functionality. However, it's important to note that more advanced algorithms, such as transformer-based models like BERT or GPT, are often used in modern chatbot systems to capture complex language structures and semantics more effectively.

# Project shots :
![img1](https://github.com/11shiv/chatbot-mental-health-conversational/assets/103626079/38e6f60b-a0e7-4c7f-8770-9fc1a29e12d2)




### Limitations and Future Improvements

- The project uses a basic bag-of-words approach, which may not handle complex medical queries effectively.
- The intent data in `intents.json` is pre-defined, limiting the chatbot's ability to handle a wide range of medical questions.
- The responses are randomly selected from predefined intent data, which might not always provide accurate or appropriate answers.
- The model lacks the ability to understand context or nuances in questions.

To create a more useful and accurate medical chatbot, future improvements could include:

- Using more advanced natural language processing techniques, like word embeddings or transformer models.
- Accessing reliable medical databases or APIs to provide accurate and up-to-date information.
- Implementing context-awareness to better understand and respond to user queries.
- Ensuring that the responses provided are accurate, evidence-based, and consider the ethical implications of medical advice.

## conclusion : 
while the provided code snippet demonstrates the basic structure of a medical chatbot, building a truly effective and reliable medical chatbot requires more advanced techniques and a thorough understanding of medical information and language processing.
