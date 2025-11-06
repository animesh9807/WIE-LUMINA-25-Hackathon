import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random

data = [
    # Original 21
    ("Get free money now", 'spam'),
    ("Click this link to win a prize", 'spam'),
    ("Free Course", 'spam'),
    ("Your device has been logged in on another device", 'ham'),
    ("Buy cheap Vbucks online from my site", 'spam'),
    ("Join my Masterclass", 'ham'),
    ("Hello, how are you today?", 'ham'),
    ("Meeting scheduled for tomorrow at 10 AM", 'ham'),
    ("I visited Switzerland. It was fun", 'ham'),
    ("Please find the attached report document", 'ham'),
    ("What's for dinner tonight?", 'ham'),
    ("Your project update is due Friday", 'ham'),
    ("Exclusive offer:50% OFF!", 'spam'),
    ("Festive season offers. Discount on our website", 'spam'),
    ("CHEAP CLOTHES AVAILABLE AT OUR STORE", 'spam'),
    ("Reminder about the meeting scheduled for tomorrow at 10 AM", 'ham'),
    ("Join my workshop/masterclass", 'ham'),
    ("Exclusive offer to buy a bugatti for free", 'spam'),
    ("Hot Single Moms in your Area", 'spam'),
    ("Free Prize Won Lottery Cas Money Bonus Congratulations $$$", 'spam'),
    ("Urgent Act now Limited time Expires Account suspended Warning Security alert", 'ham'),
    
    # Added 79 more
    ("Congratulations! You won a free iPhone", 'spam'),
    ("Limited time offer, claim your free gift card", 'spam'),
    ("Work from home and earn $5000 weekly", 'spam'),
    ("Your subscription will expire soon", 'ham'),
    ("Team meeting at 3 PM today", 'ham'),
    ("Update your account information now", 'spam'),
    ("Lunch at the new Italian restaurant?", 'ham'),
    ("Don't miss our Black Friday deals!", 'spam'),
    ("Join our free webinar on data science", 'spam'),
    ("Can you review the document by tonight?", 'ham'),
    ("Reminder: submit your timesheet", 'ham'),
    ("Your parcel has been shipped", 'ham'),
    ("Unlock your free trial of premium software", 'spam'),
    ("Earn money online with zero investment", 'spam'),
    ("Happy birthday! Have a great day!", 'ham'),
    ("Final notice: account suspension imminent", 'spam'),
    ("Let's catch up this weekend", 'ham'),
    ("Exclusive discount for our loyal customers", 'spam'),
    ("Your invoice for September is attached", 'ham'),
    ("Click here to reset your password", 'spam'),
    ("Join our mentorship program for free", 'spam'),
    ("Can you send me the slides from yesterday?", 'ham'),
    ("Free online course for beginners", 'spam'),
    ("Project deadline is extended to next week", 'ham'),
    ("Your bank account statement is ready", 'ham'),
    ("Limited stock available, buy now!", 'spam'),
    ("Team outing scheduled for Friday", 'ham'),
    ("Sign up for our newsletter and win prizes", 'spam'),
    ("Your flight itinerary for tomorrow", 'ham'),
    ("Special promotion: 70% off selected items", 'spam'),
    ("Please approve the attached leave request", 'ham'),
    ("You have been selected for a cash reward", 'spam'),
    ("Reminder: doctor's appointment at 5 PM", 'ham'),
    ("Get your free trial subscription now", 'spam'),
    ("Can you attend the client call today?", 'ham'),
    ("Urgent: verify your email account", 'spam'),
    ("Looking forward to our coffee meeting", 'ham'),
    ("Limited offer: Buy 1 get 1 free", 'spam'),
    ("Check out our new product launch", 'spam'),
    ("Please review the code changes", 'ham'),
    ("Win a brand new car! Click here", 'spam'),
    ("Your tax documents are attached", 'ham'),
    ("Special discount for new customers", 'spam'),
    ("Team briefing at 11 AM tomorrow", 'ham'),
    ("Your membership is expiring soon", 'spam'),
    ("Don't forget to submit your report", 'ham'),
    ("Earn rewards by completing surveys", 'spam'),
    ("Can you join the Zoom call?", 'ham'),
    ("Flash sale: Get 50% off today only", 'spam'),
    ("Your subscription has been renewed", 'ham'),
    ("Win cash prizes instantly", 'spam'),
    ("Birthday wishes from the team!", 'ham'),
    ("Sign up now to get exclusive access", 'spam'),
    ("The presentation deck is ready for review", 'ham'),
    ("Congratulations, you've been selected!", 'spam'),
    ("Meeting agenda for tomorrow", 'ham'),
    ("Download free templates for your project", 'spam'),
    ("Are you attending the workshop?", 'ham'),
    ("Urgent: Claim your reward before it expires", 'spam'),
    ("Let's finalize the project milestones", 'ham'),
    ("Special giveaway for our subscribers", 'spam'),
    ("Can you check the server logs?", 'ham'),
    ("Limited time bonus: Act now!", 'spam'),
    ("Reminder: team standup at 10 AM", 'ham'),
    ("Free online tickets for concert", 'spam'),
    ("Your performance review is scheduled", 'ham'),
    ("Exclusive VIP access to our website", 'spam'),
    ("Please sign the NDA document", 'ham'),
    ("Cashback offer: Shop today", 'spam'),
    ("Client feedback received for your project", 'ham'),
    ("Act now to claim your reward", 'spam'),
    ("Your order has been delivered", 'ham'),
    ("Discount coupons for new arrivals", 'spam'),
    ("Check the analytics dashboard", 'ham'),
    ("Win a trip to Bali", 'spam'),
    ("Team lunch next Wednesday", 'ham'),
    ("Register for free coding workshop", 'spam'),
    ("Budget meeting scheduled", 'ham'),
    ("Get free access to premium tools", 'spam'),
    ("Your subscription payment is due", 'ham'),
    ("Congratulations, you won a voucher!", 'spam'),
    ("Can you review the presentation slides?", 'ham'),
    ("Flash offer: Limited stock available", 'spam'),
    ("Submit your project proposal by Monday", 'ham'),
    ("Win an iPad instantly", 'spam'),
    ("Team check-in call at 2 PM", 'ham'),
    ("Exclusive access to our new software", 'spam'),
    ("Please confirm attendance for the seminar", 'ham'),
    ("Special prize for top participants", 'spam'),
    ("The quarterly report is ready", 'ham'),
    ("Claim your free ebook now", 'spam'),
    ("Join the client meeting at 4 PM", 'ham'),
    ("Offer valid only for today", 'spam'),
    ("Please review the latest code commit", 'ham'),
    ("Win gift cards up to $1000", 'spam'),
    ("Monthly performance summary attached", 'ham'),
    ("Shut the fuck up and give me your password", 'spam'),
    ("You worthless piece of shit, click this link", 'spam'),
    ("Fuck you! Earn money fast here", 'spam'),
    ("If you don't pay, we'll come after you", 'spam'),
    ("Give us access or else", 'spam'),
    ("I will expose your secrets, click here to settle", 'spam'),
    ("Bastard! Win free cash now", 'spam'),
    ("Send nudes or get exposed", 'spam'),
    ("Drop dead, loser. Claim your prize now", 'spam'),
    ("You are an idiot — sign up to learn how to stop being one", 'spam'),
    ("I'll find you if you ignore this message", 'spam'),
    ("Pay the fee or face consequences", 'spam'),
    ("You have been targeted — act now to protect yourself", 'spam'),
    ("Die motherfucker, click this link", 'spam'),
    ("Eat shit and click here for a gift", 'spam'),
    ("We will ruin your reputation unless you respond", 'spam'),
    ("You piece of trash, claim your free reward", 'spam'),
    ("Get fucked — limited time offer", 'spam'),
    ("We know where you live. Pay up", 'spam'),
    ("Suck my dick to get access", 'spam'),
    ("You pathetic loser, redeem your prize now", 'spam'),
    ("I hope you die, click to see why", 'spam'),
    ("Die you cunt, win free vouchers", 'spam'),
    ("Send money now or suffer the consequences", 'spam'),
    ("I will beat you if you don't call", 'spam'),
    ("Rot in hell, click here for a surprise", 'spam'),
    ("You're worthless — learn how to make money fast", 'spam'),
    ("We hacked your account, send BTC to unlock", 'spam'),
    ("Go fuck yourself and enter your card details", 'spam'),
    ("We will leak your photos unless you pay", 'spam')
]

spam_templates = [
    "Win a free {} now!",
    "Click here to claim your {}",
    "Limited offer: Get {} today",
    "Act fast to earn {}",
    "You are selected for {}",
    "Free {} available for a short time",
    "Urgent: Verify your account to get {}",
    "Pay {} or face consequences",
    "Don't miss this chance to get {}",
    "Congratulations! You earned {}"
]

ham_templates = [
    "Please review the {} document",
    "Meeting scheduled for {}",
    "Lunch at {}?",
    "Reminder: {} due tomorrow",
    "Happy birthday {}!",
    "Can you check the {}?",
    "Your {} is ready for review",
    "Team outing planned on {}",
    "Your subscription for {} is renewed",
    "Please approve the {} request"
]

aggressive_templates = [
    "Shut the {} up and {}",
    "You worthless {}! Click {}",
    "If you don't {}, we will {}",
    "Pay {} now or face {}",
    "We will ruin your {} unless you {}",
    "Rot in {}! Claim your {}",
    "You're {}! Win {}",
    "Send {} or get {}",
    "Die {}! Click {}",
    "Go {} yourself and {}"
]

spam_words = ["iPhone", "cash", "gift card", "$1000", "Bitcoin", "lottery", "free course", "prize", "voucher", "discount"]
ham_words = ["project", "slides", "meeting", "report", "presentation", "invoice", "appointment", "dashboard", "review", "deadline"]
aggressive_words = ["fuck", "shit", "loser", "cunt", "die", "bastard", "suck", "pathetic", "worthless", "moron"]
random_places = ["now", "today", "tomorrow", "immediately", "instantly", "this week"]

new_data = []

for _ in range(100):
    new_data.append((random.choice(spam_templates).format(random.choice(spam_words)), 'spam'))
for _ in range(100):
    new_data.append((random.choice(ham_templates).format(random.choice(ham_words)), 'ham'))
for _ in range(50):
    new_data.append((random.choice(aggressive_templates).format(random.choice(aggressive_words), random.choice(aggressive_words)), 'spam'))

# Shuffle to mix spam/ham
random.shuffle(new_data)


data.extend(new_data)

X_train = [message for message, label in data]
y_train = [label for message, label in data]



spam_classifier_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

spam_classifier_pipeline.fit(X_train, y_train)

# Streamlit UI

st.title("Spam vs Ham Classifier 📨")
st.markdown("Type a message below and see if it's **SPAM** or **HAM**.")


user_input = st.text_area("Enter your message here:")

if st.button("Classify Message"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        
        prediction = spam_classifier_pipeline.predict([user_input])[0]
        st.write(f"**Prediction:** {prediction.upper()}")

        # Predict probabilities
        try:
            probabilities = spam_classifier_pipeline.predict_proba([user_input])[0]
            st.write(f"**Probability:** HAM: {probabilities[0]:.2f}, SPAM: {probabilities[1]:.2f}")
        except AttributeError:
            st.write("(Classifier does not support probability scores)")

        # Action suggestion
        if prediction == 'spam':
            st.error("Action: Flag comment as spam or hold for moderation.")
        else:
            st.success("Action: Approve comment and post.")

