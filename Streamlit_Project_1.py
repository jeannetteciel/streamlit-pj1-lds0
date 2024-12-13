
# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import regex
import regex
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from underthesea import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

df = pd.read_csv('data_for_model.csv')

### For New Prediction
# Chuyển nội dung về chữ thường trước khi remove duplicate
df.processed_cmt = df.processed_cmt.str.lower()

df['processed_cmt'] = df['processed_cmt'].fillna('')
X_train, X_test, y_train, y_test = train_test_split(df['processed_cmt'], df['rating_group'], test_size=0.3, random_state=1)
vectorize = TfidfVectorizer()
X_train_tfidf = vectorize.fit_transform(X_train)
X_test_tfidf = vectorize.transform(X_test)
rus = RandomUnderSampler(random_state=0)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_tfidf, y_train)
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
result = model.predict(X_test_tfidf)

train_acc = model.score(X_train_resampled, y_train_resampled)
test_acc = model.score(X_test_tfidf, y_test)

rp =classification_report(y_test, result)

joblib.dump(model, 'logisticregression.joblib')
loaded_model = joblib.load('logisticregression.joblib')

###

### For Product Analysis
# Lấy 20 sản phẩm
random_products = df.head(n=1000)
# print(random_products)

st.session_state.random_products = random_products

# Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
if 'selected_ma_san_pham' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
    st.session_state.selected_ma_san_pham = None

# Tạo set các stopwords:
stopwords = set()
f = open(r'vietnamese-stopwords.txt', "r", encoding='utf-8')
for line in f:
    word = f.readline()
    stopwords.add(word.replace('\n',''))
f.close()

list_of_words = ['và', 'một', 'của', 'có', 'đó', 'rất', 'nào', 'được',
                'khi', 'thể', 'sự', 'tính', 'trong','cũng','cùng','cho','hay','chỉ', 'hasaki', 'cực', 'ghế','mặt']
for word in list_of_words:
    stopwords.add(word)

product_stop_words_file = pd.read_csv('product_stop_words.csv')
product_stop_words = product_stop_words_file['words'].to_list()
for word in product_stop_words:
   stopwords.add(word)

pos = pd.read_csv('positive_words.csv')
positive_words = pos['positive_words'].to_list()
positive_words_u = []
for i in positive_words:
  a = i.replace(" ", "_")
  positive_words_u = positive_words_u + [a]

neg = pd.read_csv('negative_words.csv')
negative_words = neg['negative_words'].to_list()
negative_words_u = []
for i in negative_words:
  a = i.replace(" ", "_")
  negative_words_u = negative_words_u + [a]

def process_comment(comment):
    # Tokenize các từ
    tokens = word_tokenize(comment)
    # Loại bỏ stop words và các kí tự không cần thiết
    stop_words = stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Xử lý các từ trong comment và tính toán các từ thường gặp
def cmt_extract (df, comment_column):
  all_words = []
  positive_list = []
  negative_list = []
  for index, row in df.iterrows():
      comment = row[comment_column]
      tokens = process_comment(comment)
      all_words.extend(tokens)
      if row['rating_group'] == 'positive' and any(token for token in tokens):
          positive_list.extend(tokens)
      elif row['rating_group'] == 'negative' and any(token for token in tokens):
          negative_list.extend(tokens)
  return all_words, positive_list, negative_list


# App Design
###### Giao diện Streamlit ######
st.image('hasakibanner.png', use_container_width=True)
st.title('Project 1 - Hasaki Sentiment Analysis')

menu = ['Project Executive', 'New Prediction', 'Product Analysis']
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Hà Thúy An & Trương Thanh Tuyền""")
st.sidebar.write("""#### Giảng viên hướng dẫn:
                (Cô) Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")

if choice == 'Project Executive':
  st.subheader("Overal description of customers' feedback on Hasaki skincare products")
  st.write('Research is conducted on ', len(df['noi_dung_binh_luan']), ' comments covering ', len(df['ma_san_pham'].unique()), ' products.')
  st.write('## Product Average Rating')
  st.write('Most of product averagely score over 4-star, showing that Hasaki is providing reliable and standard-meeting products')
  st.image('rating_outliers.png', use_container_width=True)
  st.write('## Comment Analysis')
  st.write('### Rating Distribution')
  st.write(format(len(df[df['so_sao'] == 5]['so_sao'])/len(df['so_sao']), '.0%'), ' rating is 5-star showing overall positive feedback on our products.')
  st.image('rating_distribution.png', use_container_width=True)
  st.write("### Positive ratio in comments' word")
  st.write('There is positive correlation in positive ratio in comments and rating. Furthermore, higher rating less fluctuation shows that people often have a mix of positive and negative words even in low-rating comments.')
  st.image('positive_ratio.png', use_container_width=True)
  st.code(df.groupby('so_sao')[['positive_ratio']].mean())
  st.write('Follow up above statistics, for feedback with 4-star, about or under 50% words are negative reviews. Thus, we decided to count all 4-star and below as negative and only 5-star reviews are positive, with the purpose of addressing every flaws and providing the best service.')
  st.write("### Most common words in positive comments")
  st.image('positive.png', use_container_width=True) 
  st.write("### Most common words in negative comments") 
  st.image('negative.png', use_container_width=True)  
  st.write('## Classification Report')
  st.write('The model is build by logistic regression with below result:')
  st.code(rp)
elif choice == 'New Prediction':
  st.subheader('Apply Logistic Regression Model to predict a new comment negative or positive')
  st.write(" Input or Load new comments")
  flag = False
  lines = None
  type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
  if type == 'Upload':
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'txt'])
    if uploaded_file is not None:
            lines = pd.read_csv(uploaded_file, header=None)
            st.dataframe(lines)
            lines = lines[0]
            flag = True
  if type=="Input":
      content = st.text_area(label="Input your content:")
      if content!="":
          lines = content.split('\n')  # Split the input by newline characters
          lines = [line.strip() for line in lines if line.strip()] # Remove empty lines
          flag = True

  if flag:
    st.write("Content:")
    if len(lines)>0:
        st.code(lines)

        # Convert lines to a list if it's a NumPy array or Series
        if isinstance(lines, (np.ndarray, pd.Series)):
            lines = lines.tolist()

        # Vectorize and predict for each line
        new_comment_vectorized = vectorize.transform(lines)
        prediction = loaded_model.predict(new_comment_vectorized)

        # Create a DataFrame for the table
        data = {'Comment': lines, 'Prediction': prediction}
        df_predictions = pd.DataFrame(data)

        # Display the table using st.dataframe
        st.dataframe(df_predictions)

elif choice == 'Product Analysis':
  st.subheader('Get to know your product')
  # Theo cách cho người dùng chọn sản phẩm từ dropdown
  # Get unique product names and codes
  unique_products = st.session_state.random_products[['ten_san_pham', 'ma_san_pham']].drop_duplicates(subset=['ten_san_pham'])
  # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
  product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in unique_products.iterrows()]
  # st.session_state.random_products
  # Tạo một dropdown với options là các tuple này
  selected_product = st.selectbox(
    "Chọn sản phẩm",
    options=product_options,
    index = None,
    placeholder = "Choose an option",
    label_visibility="visible",
    format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
  )
  if selected_product:
    # Display the selected product
    st.write("Bạn đã chọn:", selected_product[0])
    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]
    product_code = st.session_state.selected_ma_san_pham
    # Lọc dữ liệu cho sản phẩm có mã tương ứng
    product_data = df[df['ma_san_pham'] == product_code]

    st.write(product_data[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh', 'gia_ban', 'gia_goc', 'phan_loai'
                            , 'ma_khach_hang', 'noi_dung_binh_luan', 'ngay_binh_luan', 'so_sao', 'rating_group']]
                            .rename(columns = {'ma_san_pham':'Mã sản phẩm', 'ten_san_pham':'Tên sản phẩm'
                            ,'diem_trung_binh':'Điểm trung bình', 'gia_ban':'Giá bán'
                            , 'gia_goc':'Giá gốc', 'phan_loai':'Phân loại'
                            , 'ma_khach_hang':'Mã khách hàng', 'noi_dung_binh_luan':'Nội dung bình luận'
                            , 'ngay_binh_luan':'Ngày bình luận', 'so_sao':'Rating', 'rating_group':'Phân loại đánh giá'}))
    
    all_words, positive_list, negative_list = cmt_extract(product_data, 'processed_cmt')

    # Số lượt đánh giá
    num_reviews = len(product_data)

    # Số lượng đánh giá của mỗi loại (tích cực - tiêu cực - trung bình)
    positive_reviews = len(product_data[product_data['rating_group'] == 'positive'])
    negative_reviews = len(product_data[product_data['rating_group'] == 'negative'])

    # Các từ thường gặp trong comment
    freq_dist = FreqDist(all_words)
    freq_dist_positive = FreqDist(positive_list)
    freq_dist_negative = FreqDist(negative_list)

    # Các từ thường xuất hiện phần đánh giá tích cực và tiêu cực
    common_positive_words = freq_dist_positive.most_common(10)
    common_negative_words = freq_dist_negative.most_common(10)

    positive_comment_words =' '.join(positive_list)
    negative_comment_words =' '.join(negative_list)

    cmt = product_data[['noi_dung_binh_luan', 'rating_group']]
    if st.session_state.selected_ma_san_pham:
      st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
      # Hiển thị thông tin sản phẩm được chọn
      selected_product = df[df['ma_san_pham'] == st.session_state.selected_ma_san_pham]

      if not selected_product.empty:
        st.write('## Đánh giá sản phẩm')
        st.write('### ', selected_product['ten_san_pham'].values[0])
        st.write('Số lượt đánh giá:', num_reviews)
        st.write('Điểm trung bình:', selected_product['diem_trung_binh'].values[0])
        st.write('Số lượng đánh giá tích cực:', positive_reviews)
        st.write('Số lượng đánh giá tiêu cực:', negative_reviews)

        st.write('#### Từ thường xuất hiện ở đánh giá tích cực: ')
        if len(positive_comment_words) > 0:
          # Vẽ wordclouds
          wc_like=WordCloud(background_color='white', max_words=100, stopwords=stopwords)
          wc_like.generate(positive_comment_words)
          plt.figure(figsize=(10, 12))
          plt.imshow(wc_like, interpolation='bilinear')
          plt.axis('off')
          plt.show()
          st.pyplot(plt)
        else:
          st.write('Không có bình luận tích cực')

        st.write('#### Từ thường xuất hiện ở đánh giá tiêu cực: ')
        if len(negative_comment_words) >0:
          # Vẽ wordclouds
          wc_like=WordCloud(background_color='white', max_words=100, stopwords=stopwords)
          wc_like.generate(negative_comment_words)
          plt.figure(figsize=(10, 12))
          plt.imshow(wc_like, interpolation='bilinear')
          plt.axis('off')
          plt.show()
          st.pyplot(plt)
        else:
          st.write('Không có bình luận tiêu cực')
      else:
        st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")

