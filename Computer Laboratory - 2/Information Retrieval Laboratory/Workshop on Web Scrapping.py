#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries Required

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import concurrent.futures


# In[5]:


soup = BeautifulSoup(response.text,'html.parser')


# In[6]:


url='https://www.google.com'


# In[7]:


response = requests.get(url)


# In[8]:


if response.status_code == 200:
    print(response.text)
else:
    print(f"failed to retrieve : {response.status_code}")


# In[9]:


title = soup.title.string
print("Page Title", title)


# In[10]:


First_para = soup.find('p')
if First_para:
        print("First Para", First_para.text)
else:
    print("No <p> element found on page.")


# # Get Hyperlinks

# In[11]:


for link in soup.find_all('a'):
    print("hyperlink",link.get('href'))


# # Download PPT from URL

# In[12]:


url ='https://www.thehindu.com/news/national/coronavirus-live-updates-may-29-2021/article34672944.ece?homepage=true'


# In[13]:


page = requests.get(url)


# In[14]:


page


# In[15]:


soup = BeautifulSoup(page.content,'html.parser')


# In[16]:


img_tag = soup.find('source')
img_tag
                   


# In[17]:


img_tag['srcset']


# In[18]:


img_url = img_tag['srcset']


# In[19]:


image = requests.get(img_url)


# In[20]:


with open('image.jpg','wb') as file:
    for chunk in image.iter_content(chunk_size=1024):
        file.write(chunk)


# In[21]:


ppt = requests.get('http://www.howtowebscrape.com/examples/media/images/SampleSlides.pptx')


# In[22]:


with open('sample.pptx','wb') as file:
    for chunk in image.iter_content(chunk_size=1024):
        file.write(chunk)


# # Download video from URL

# In[23]:


video = requests.get('http://www.howtowebscrape.com/examples/media/images/BigRabbit.mp4')


# In[24]:


with open('Bigrabbit.mp4','wb') as file:
    for chunk in video.iter_content(chunk_size=1024):
        file.write(chunk)


# In[25]:


url = "https://www.imdb.com/chart/top/"


# In[26]:


HEADERS = {'User-Agent': 'Mozilla/5.0 (ipad; CPU OS12_2 like Mac OS X) Applewebkit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}


# In[27]:


page = requests.get(url, headers=HEADERS)
page


# In[28]:


soup = BeautifulSoup(page.content, "html.parser")


# In[29]:


scrapped_movie = soup.find_all('h3', class_='ipc-title__text')
scrapped_movie 


# In[35]:


movies = []

for movie in scrapped_movie:
    movie = movie.get_text().replace('\n','')
    movie = movie.strip(" ")
    movies.append(movie)
movies


# In[36]:


scraped_ratings = soup.find_all('span', class_= 'ipc-rating-star ipc-rating-star--base ipc-rating-star--imdb ratingGroup--imdb-rating')
scraped_ratings 


# In[37]:


ratings = []
for rating in scraped_ratings:
        rating = rating.get_text().replace('\n','')
        ratings.append(rating)
ratings


# In[41]:


# Assuming 'movies' and 'ratings' are your original lists
movies = ["Movie 1", "Movie 2", "Movie 3", ...]  # Your list of movie names
ratings = [4.5, 3.0, None, ...]  # Your list of ratings with some missing values

# Make sure both lists have the same length by adding NaN for missing ratings
max_length = max(len(movies), len(ratings))
movies += [''] * (max_length - len(movies))
ratings += [np.nan] * (max_length - len(ratings))



data = pd.DataFrame()
data['Movie Name']= movies
data['Ratings'] = ratings
data.head()


# In[ ]:





# In[42]:


data = pd.DataFrame({'Movie Name': movies, 'Ratings': ratings})


data.head()


# # Afternoon Session

# In[43]:


import requests
def crawl_web(seed_url, max_pages):
    pages_visited= 0
    url_queue= [seed_url]
    
    while pages_visited < max_pages and url_queue:
        url = url_queue.pop(0)
        
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text,'html.parser')
                print(soup.title.string)
            
                links = [a['href']for a in soup.find_all('a',href=True)]
            
                for link in links:
                    url_queue.append(link)
                
                
            pages_visited +=1
            
        except Exception as e:
            print (f"Errors:{e}")
   
crawl_web("https://google.com",max_pages=8)


# In[44]:


def calculate_pagerank(adjacency_matrix, dumping_factor= 0.85, max_iterarions=100, tolerance= 1e-6):
    num_nodes= adjacency_matrix.shape[0]
    initial_pagerank = np.ones(num_nodes) / num_modes
    pagerank = initial_pagerank.copy()
    
    for iteration in range(max_iterationns):
        new_pagerank = np.zeroes(num_nodes)
        for i in range (num_nodes):
            for j in range (num_nodes):
                if adjacency_matrix[j,i] ==1:
                    outgoing_links = np.sum(adjacency_matrix[j])
                    new_pagerank[i]+= pagerank[j] / outgoing_link
        new_pagerank= (1-damping_factor) / num_nodes + damping_factor * new_pagerank
    
        if np.linalg.norm(new_pagerank - pagerank) < tolerance :
            break
        pagerank = new_pagerank  
    
    return pagerank
    
                  


# In[45]:


import numpy as np

def calculate_pagerank(adjacency_matrix, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_nodes = adjacency_matrix.shape[0]
    initial_pagerank = np.ones(num_nodes) / num_nodes
    pagerank = initial_pagerank.copy()

    for iteration in range(max_iterations):
        new_pagerank = np.zeros(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_matrix[j, i] == 1:
                    outgoing_links = np.sum(adjacency_matrix[j])
                    new_pagerank[i] += pagerank[j] / outgoing_links
        new_pagerank = (1 - damping_factor) / num_nodes + damping_factor * new_pagerank

        if np.linalg.norm(new_pagerank - pagerank) < tolerance:
            break

        pagerank = new_pagerank

    return pagerank

# Example usage:
# Create a sample adjacency matrix representing a directed graph
adjacency_matrix = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])

# Calculate PageRank scores for the graph
pagerank_scores = calculate_pagerank(adjacency_matrix)

# Print the PageRank scores for each node
for i, score in enumerate(pagerank_scores):
    print(f"Node {i + 1}: {score:.4f}")


# In[46]:


# HITS 
# TRUST RANK - university gov 
# Propogatte measures 


# In[47]:


"""analysis of all links
crawl link on trust 
analysis of links trusted to display"""



# In[ ]:





# In[48]:


import requests
import concurrent.futures
from bs4 import BeautifulSoup

# Function to scrape quotes from a URL
def scrape_quotes(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract and print quotes from the webpage (modify this part)
            quotes = soup.find_all('span', class_='text')
            for quote in quotes:
                print(quote.text)
            print(f"Processed {url}")

        else:
            print(f"Failed to retrieve data from {url}")

    except Exception as e:
        print(f"An error occurred while processing {url}: {str(e)}")

# List of URLs to crawl (replace with your own URLs)
urls_to_crawl = [
    'http://quotes.toscrape.com/page/1/',
    'http://quotes.toscrape.com/page/2/',
    'http://quotes.toscrape.com/page/3/',
    'http://quotes.toscrape.com/page/4/',
    'http://quotes.toscrape.com/page/5/'
    # Add more URLs as needed
]

# Number of concurrent threads to use for crawling
num_threads = 4  # Adjust as needed

# Create a ThreadPoolExecutor with the specified number of threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit crawling tasks for each URL
    futures = [executor.submit(scrape_quotes, url) for url in urls_to_crawl]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("Crawling completed.")


# In[ ]:





# In[ ]:




