# Statt-Testing
Welcome to the **Policy Chatbot**! This chatbot allows you to ask questions related to **cybersecurity, IT policies, data privacy, and compliance regulations.** All information is collected from U.S. Department of Commerce (DOC) website. Follow the steps below to set up and run the chatbot.

## 1️⃣First Thing to Do--Clone the Repository
To start, **clone this repository** to your local machine:
```bash
git clone https://github.com/Vika77-kuromi/Statt-Testing.git
cd Statt-Testing
```

## 2️⃣ Create a virtual environment to test this application. (Optional)
```bash
python3 -m venv venv 
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate      # For Windows
```

## 3️⃣Setting Up OpenAI API Key
This chatbox requires an OpenAI API key to function. Follow these steps:

1. Users need to create a new file named ".env" in the same directory where the chatbot code is located.
   ```bash
   touch .env
   nano .env
   ```
2. Add the following line to the .env file:
   OPENAI_API_KEY=your-api-key-here
3. Replace "your-api-key-here" with your actual OpenAI API key.
4. Save the file (CTRL+X, then Y, then ENTER).

## 4️⃣Installing Dependencies
Install all the required packages for all the codes to be able to run
```bash
pip install -r requirements.txt
```
If you're using a virtual environment, run:
```bash
pip install --no-cache-dir -r requirements.txt
```

## 5️⃣ The Final Step - Run the Chatbot !!
After setting up your API key, run the chatbot with:
   ```bash
   python policytesting1.py
   ```

## Here are some sample questions to ask 👀:
*What are the key principles of the DOC Enterprise Cybersecurity Policy?*  
*How does the DOC handle cybersecurity risks?*  
*How does the DOC define IT governance?*  
*When does the DOC plan to fully implement IPv6?* 
*How does the DOC handle IT product end-of-life management?*  
*What are the maintenance requirements for DOC IT assets?* 
*What are the approval processes for new IT policies?*

Enjoy it☺️!


