import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset


st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

@st.cache(allow_output_mutation=True)
def load_data():
    # df_original = pd.read_csv("./data/github_dataset.csv")
    # df = df_original.copy()
    # df.drop_duplicates(subset='name', inplace=True)
    # return df
    repos = load_dataset('abhishekbora09/github_repositories', split='train')
    df_hf = repos.to_pandas()
    df = df_hf.copy()
    df.drop_duplicates(subset='name', inplace=True)
    return df

df = load_data()

st.sidebar.header('`GitHub Repository Dashboard`')

#Convert 'Created_at' column to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
# Extract year from 'Created_at' column
df['Year'] = df['created_at'].dt.year

#Get the count of each language
language_counts = df['primary_language'].value_counts()
# Select the top 20 languages
top_20_languages = language_counts.head(20)
# Convert the top 20 languages to a DataFrame
top_20Language_df = pd.DataFrame({'primary_language': top_20_languages.index, 'Count': top_20_languages.values})
language_option = top_20Language_df['primary_language'].unique().tolist()


st.markdown('# GITHUB REPOSITORY DATA ANALYSIS')
st.write('-------------------------------------------------------')
#---------------------------------------------------------------------------------------------------------

#GLIMPSE OF THE DATASET
st.markdown("## Glimpse of the Dataset")
st.write(df.head(5))
# st.write(df.tail(5))
# st.markdown("### High Level Stats about the dataset")
# st.write(df_original.describe())
st.write('-------------------------------------------------------')

#---------------------------------------------------------------------------------------------------------
# Metrics

st.markdown('## IMPORTANT METRICS')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Most Popular Language", "Javascript")
col2.metric("Most Popular Repository", "freeCodeCamp")
col3.metric("Most Used Licence", "MIT Licence")
col4.metric("Most No. of Repos Created in", "2020")
st.write('-------------------------------------------------------')

#---------------------------------------------------------------------------------------------------------
st.markdown("## REPOSITORY ANALYSIS")

# TOP REPOSITORIES

# with st.form(key = 'Form_repoSelectBox'):
top_repo_input = st.selectbox('Top 10 Repositories Based On', ('Star Count', 'Forks', 'Pull Requests', 'Watchers', 'Commits'))
    # submit_repoSelectBox = st.form_submit_button(label = 'Plot the Chart')
st.markdown("### Top 10 Repositories based on number of " + top_repo_input)

if (top_repo_input == "Star Count"):
    c1, c2 = st.columns((7,3))
    with c1:
        top_stars = df.sort_values(by='stars_count', ascending=False)[:10].reset_index(drop=True)
        top_stars.index += 1
        fig = px.bar(top_stars, x='name', y='stars_count', 
        labels={'name': 'Repository name ', 'stars_count': 'Number of Stars '},
        title='10 Most Popular Repositories')
        st.write(fig)
    with c2:
        st.write(top_stars[['name', 'stars_count']])

elif(top_repo_input == "Pull Requests"):
    c1, c2 = st.columns((7,3))
    with c1:
        top_updates = df.sort_values(by='pull_requests', ascending=False)[:10].reset_index(drop=True)
        top_updates.index += 1
        fig = px.bar(top_updates, x='name', y='pull_requests', 
        labels={'name': 'Repository names', 'pull_requests': 'Number of Pull Requests'},
        title='10 Most Updated Repositories')
        st.write(fig)
    with c2:
        st.write(top_updates[['name', 'pull_requests']].rename_axis('Rank'))

elif(top_repo_input == "Forks"):
    c1, c2 = st.columns((7,3))
    with c1:
        top_forks = df.sort_values(by='forks_count', ascending=False)[:10].reset_index(drop=True)
        top_forks.index += 1
        fig = px.bar(top_forks, x='name', y='forks_count', 
        labels={'name': 'Repository names', 'forks_count': 'Number of Forks'},
        title='10 Most Forked Repositories')
        st.write(fig)
    with c2:
        st.write(top_forks[['name', 'forks_count']])

elif(top_repo_input == "Watchers"):
    c1, c2 = st.columns((7,3))
    with c1:
        top_watchers = df.sort_values(by='watchers', ascending=False)[:10].reset_index(drop=True)
        top_watchers.index += 1
        fig = px.bar(top_watchers, x='name', y='watchers', 
        labels={'name': 'Repository names', 'watchers': 'Number of Forks'},
        title='10 Most Watched Repositories')
        st.write(fig)
    with c2:
        st.write(top_watchers[['name', 'watchers']])

elif(top_repo_input == "Commits"):
    c1, c2 = st.columns((7,3))
    with c1:
        top_commits = df.sort_values(by='commit_count', ascending=False)[:10].reset_index(drop=True)
        top_commits.index += 1
        fig = px.bar(top_commits, x='name', y='commit_count', 
        labels={'name': 'Repository names', 'commit_count': 'Number of Forks'},
        title='10 Most Forked Repositories')
        st.write(fig)
    with c2:
        st.write(top_commits[['name', 'commit_count']])
    


#---------------------------------------------------------------------------------------------------------
#YEAR WISE NO. OF REPOSITORIES CREATED

@st.cache_data
def repo_per_year():
    st.markdown("#### Year Wise No. of Repositories created Trend")
    # Count repositories per year
    repo_count_per_year = df['Year'].value_counts().sort_index()

    c1, c2 = st.columns((7,3))
    with c1:
        # Create a line plot
        fig = px.line(x=repo_count_per_year.index, y=repo_count_per_year.values, 
                    labels={'x': 'Year', 'y': 'Number of Repositories'},
                    title='Number of Repositories Created per Year')
        fig.update_xaxes(type='category', title='Year')
        fig.update_layout(showlegend=False)
        st.write(fig)
    with c2:
        st.write("The number of repositories created was increasing every year till the year 2020 which had the most no. of Repositories created. We see a dip in the no. of repositories created since then.")

repo_per_year()
st.write('-------------------------------------------------------')

#---------------------------------------------------------------------------------------------------------
# ANALYSIS BASED ON THE LANGUAGE

st.markdown("## LANGUAGE ANALYSIS")

# Top 10 Used Languages in Bar Plot

@st.cache_data
def top10_used_language():
    new_df = df.copy()
    new_df = df.dropna(subset=['languages_used'])
    new_df['languages_used'] = new_df['languages_used'].apply(lambda x: x.strip('[]').split(', ') if isinstance(x, str) else x)
    all_languages = [lang for sublist in new_df['languages_used'] for lang in sublist]

    # Count occurrences of each language
    language_count = {}
    for language in all_languages:
        if language in language_count:
            language_count[language] += 1
        else:
            language_count[language] = 1

    # Convert the language count dictionary to a DataFrame
    language_count_df = pd.DataFrame(list(language_count.items()), columns=['Language', 'Count'])

    top_10_Ulanguages = language_count_df.nlargest(10, 'Count').reset_index(drop=True)
    top_10_Ulanguages.index += 1

    c1, c2 = st.columns((7,3))
    with c1:
        # Create a bar plot
        fig = px.bar(top_10_Ulanguages, x='Language', y='Count', 
                    labels={'Language': 'Programming Language', 'Count': 'Usage Count'},
                    title='Top 10 Most Used Programming Languages')
        fig.update_layout(xaxis={'title': {'standoff': 10}}, yaxis={'title': {'standoff': 10}})
        st.write(fig)

    with c2:
        st.write(top_10_Ulanguages)

top10_used_language()
#---------------------------------------------------------------------------------------------------------

# Top 10 Primary Languages Bar Plot


@st.cache_data
def top10_primary_language():
    top_10_languages = df['primary_language'].value_counts().head(10)
    top_10_Planguages = pd.DataFrame({'Languages': top_10_languages.index})
    top_10_Planguages['Count'] = top_10_languages.values
    top_10_Planguages.index += 1  # Start the index from 1


    c1, c2 = st.columns((7,3))
    with c1:
        fig = px.bar(top_10_Planguages, x='Languages', y='Count',
                    labels={'x': 'Primary Language', 'y': 'Count'},
                    title='Top 10 Most Used Languages as the Primary Languages')
        fig.update_layout(xaxis={'title': {'standoff': 10}}, yaxis={'title': {'standoff': 10}})
        st.write(fig)

    with c2:
        st.write(top_10_Planguages)

top10_primary_language()
#---------------------------------------------------------------------------------------------------------

#DISTRIBUTION OF TOP LANGUAGES:

@st.cache_data
def language_distribution():
    # Select top 10 languages and group the rest as 'Others'
    top_languages = language_counts.head(10)
    other_languages_count = language_counts[10:].sum()

    # Create a new dictionary for top languages (top 10)
    top_languages_dict = {'Language': top_languages.index, 'Count': top_languages.values}

    # If there are 'Others', append it to the dictionary
    if other_languages_count > 0:
        top_languages_dict['Language'] = top_languages_dict['Language'].tolist() + ['Others']
        top_languages_dict['Count'] = list(top_languages_dict['Count']) + [other_languages_count]

    # Convert the dictionary to a DataFrame
    top_languages_df = pd.DataFrame(top_languages_dict)

    # Plotting the pie chart
    fig = px.pie(top_languages_df, values='Count', names='Language', title='Distribution of Top Most Used Primary Languages')
    fig.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial')
    st.write(fig)


language_distribution()
#---------------------------------------------------------------------------------------------------------

#Language Popularity based on Year
st.markdown("#### Language Trend Lines")
with st.form(key = 'Form_multiselect'):
    selected_languages = st.multiselect('Select Primary Languages', language_option)
    submit_multiselect = st.form_submit_button(label = 'Plot the Chart')
    filtered_data = df[df['primary_language'].isin(selected_languages)]
    # Group by year and language, count repositories for each year and language
    grouped_data = filtered_data.groupby(['Year', 'primary_language']).size().reset_index(name='RepoCount')

    fig = px.line(grouped_data, x='Year', y='RepoCount', color='primary_language',
                title='Number of Repositories Created by Year for Selected Languages',
                labels={'Year': 'Year', 'RepoCount': 'Number of Repositories'})

    fig.update_traces(mode='lines+markers', marker=dict(size=8), hovertemplate=None)
    fig.update_layout(
        width=1000,
        legend_title='Languages',
        legend=dict(
            title='Languages',
            orientation='v',  
            x=1.09,  
            y=0.5,  
        ),
        xaxis_title='Year',
        yaxis_title='Number of Repositories',
        xaxis=dict(
            tickmode='linear',  
            dtick=1,  
        )
    )
    st.write(fig)
#---------------------------------------------------------------------------------------------------------


#TOP 5 REPOSITORIES BASED ON THE SELECTED PRIMARY LANGUGAE
# Filtering repositories with primary language as the selected language
st.markdown("##### Repository Stats Based on the Primary Language")

selected_language = st.selectbox('Select language', language_option, index=0)

selectedLanguage_repos = df[df['primary_language'] == selected_language]

c1, c2, c3 = st.columns((3,3,3))
with c1:
    # Sorting by stars_count in descending order and getting the top 5
    top_5_selectedLanguage_repos = selectedLanguage_repos.nlargest(5, 'stars_count')[['name', 'stars_count']].reset_index(drop=True)
    top_5_selectedLanguage_repos.index += 1

    st.write("Most Popular " + selected_language +  " Repositories: ")
    st.write(top_5_selectedLanguage_repos)

with c2:
    # Sorting by pull_requests in descending order and getting the top 5
    top_5_selectedLanguage_repos = selectedLanguage_repos.nlargest(5, 'pull_requests')[['name', 'pull_requests']].reset_index(drop=True)
    top_5_selectedLanguage_repos.index += 1

    st.write("Most Updated " + selected_language +  " Repositories: ")
    st.write(top_5_selectedLanguage_repos)

with c3:
    # Sorting by commited in descending order and getting the top 5
    top_5_selectedLanguage_repos = selectedLanguage_repos.nlargest(5, 'commit_count')[['name', 'commit_count']].reset_index(drop=True)
    top_5_selectedLanguage_repos.index += 1

    st.write("Most Committed " + selected_language +  " Repositories: ")
    st.write(top_5_selectedLanguage_repos)

st.write('-------------------------------------------------------')

#---------------------------------------------------------------------------------------------------------


# LICENCE PIE CHART
st.markdown("## LICENCE ANALYSIS")

@st.cache_data
def licence_analysis():
    st.markdown("### Licence Type Stats " + top_repo_input)
    licence_df = df['licence'].value_counts()

    c1, c2 = st.columns((7,3))
    with c1:
        # Creating a Plotly pie chart
        fig = px.pie(
            values=licence_df.values,
            names=licence_df.index,
            title='Division of the licences'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.write(fig)

    with c2:
        # Filter out NULL values in 'licence' column
        licence_df = licence_df.dropna()
        licence_df = licence_df[licence_df.index != 'Other']
        top_5_licences = licence_df.head(5)

        # Get the top 5 most used licenses
        top_5_licences_with_rank = pd.DataFrame({
        'Licence': top_5_licences.index,
        'Count': top_5_licences.values,
        }, index=range(1, len(top_5_licences) + 1))
        st.write("Top 5 Most Used Licences:")
        st.write(top_5_licences_with_rank)

licence_analysis()

#---------------------------------------------------------------------------------------------------------

st.markdown('''
---
Designed with ❤️ by [Abhishek Bora](https://abhishekbora.me/).
''')
