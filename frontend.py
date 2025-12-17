import streamlit as st
import pandas as pd
import requests
import json

st.title('Provocator Detection')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        inputs = df.to_dict(orient='records')
        
        with st.spinner("Analyzing data..."):
            response = requests.post(
                url=FASTAPI_URL,
                json=inputs
            )
        if response.status_code == 200:

            # process response from backend
            result = response.json()
            predictions = result['predictions']
            topic_labels = result['topic_labels']

            # create dataframe
            df_2 = pd.DataFrame(predictions)

            # retrive topic
            topic_0 = topic_labels.get('0', 'Unknown Topic')
            topic_1 = topic_labels.get('1', 'Unknown Topic')
    
            # sort dataframe based on date
            df_2['date'] = pd.to_datetime(df_2['date'])  # Ensure date is datetime
            df_topic0 = df_2[df_2['topics'] == 0].sort_values(by='date').reset_index(drop=True)
            df_topic1 = df_2[df_2['topics'] == 1].sort_values(by='date').reset_index(drop=True)
    
            # retrieve provocator username
            prov_username0 = df_topic0.loc[0, 'username']
            prov_username1 = df_topic1.loc[0, 'username']
    
            # retrieve provocator text
            prov_text0 = df_topic0.loc[0, 'clean_text']
            prov_text1 = df_topic1.loc[0, 'clean_text']
    
            # retrieve amplifier username
            if len(df_topic0) >= 4 and len(df_topic1) >= 4:
                amp1_username0 = df_topic0.loc[1, 'username']
                amp2_username0 = df_topic0.loc[2, 'username']
                amp3_username0 = df_topic0.loc[3, 'username']
        
                amp1_username1 = df_topic1.loc[1, 'username']
                amp2_username1 = df_topic1.loc[2, 'username']
                amp3_username1 = df_topic1.loc[3, 'username']
                
                st.success("Analysis complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Topic: {topic_0}")
                    st.write(f"**Provocator:** {prov_username0}")
                    st.write(f"*Text:* {prov_text0}")
                    st.write(f"**Amplifiers:**")
                    st.write(f"1. {amp1_username0}")
                    st.write(f"2. {amp2_username0}")
                    st.write(f"3. {amp3_username0}")
                
                with col2:
                    st.subheader(f"Topic: {topic_1}")
                    st.write(f"**Provocator:** {prov_username1}")
                    st.write(f"*Text:* {prov_text1}")
                    st.write(f"**Amplifiers:**")
                    st.write(f"1. {amp1_username1}")
                    st.write(f"2. {amp2_username1}")
                    st.write(f"3. {amp3_username1}")
                
                # Optional: Show full dataframes
                with st.expander("View all data"):
                    st.write(f"### {topic_0}")
                    st.dataframe(df_topic0)
                    st.write(f"### {topic_1}")
                    st.dataframe(df_topic1)
        else:
            st.error(f"FastAPI error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to FastAPI backend. Is your server running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again with a smaller file.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")