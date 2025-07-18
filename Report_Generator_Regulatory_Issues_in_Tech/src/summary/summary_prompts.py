from openai import OpenAI


def topic_uncertainty_by_company(text_to_analyze, model, number_of_reports, api_key, topic, entity_name):
    # Create the prompt for determining market impact and magnitude   
    prompt = (
        f"""You are an expert market analyst with deep knowledge of {topic} and the Stock market, tasked with evaluating how uncertain the outcome of '{topic}' on {entity_name} is.
        
    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
    1. Based on the provided text, evaluate the uncertainty of the outcome for {entity_name} as follows:
        - High: The outcome of {topic} on {entity_name} is totally uncertain. 
        - Medium: The outcome of {topic} on {entity_name} involves some uncertainty. 
        - Low: The outcome of {topic} is on {entity_name} is highly anticipated.
        - Past: The outcome of {topic} on {entity_name} is already known. 
    2. Provide a one-sentence explanation of the uncertainty of the outcome of {topic} on {entity_name}, across all reports. 
    Important: You should focus only on elements of the input text that explicitly mention direct implication on {entity_name}'s business and ignore any predictions on the general economy or other businesses. Only use the information present in the text. Do NOT infer from outside knowledge.
    
    Output:
    Return the result as a JSON object with the following structure:
    {{"uncertainty": "High" | "Medium" | "Low" | "Past", "explanation": "Your explanation here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Asynchronous request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )

    uncertainty_dict = response.model_dump()

    try:
        uncertainty_dict = eval(uncertainty_dict['choices'][0]['message']['content'])
    except:
        uncertainty_dict = {"uncertainty": "Error", "explanation": "Error"}  # Default in case of error
    
    return uncertainty_dict


def topic_risk_by_company(text_to_analyze, model, number_of_reports, api_key, topic, entity_name):
    # Create the prompt for determining market impact and magnitude   
    prompt = (
        f"""You are an expert market analyst with deep knowledge of {topic} and the Stock market, tasked with evaluating the risks that '{topic}' involve on {entity_name} in terms of financial impact. 

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
    1. Based on the provided text, evaluate the magnitude of the risks for {entity_name} as follows:
        - High: The topic {topic} presents a high risk for {entity_name} and is expected to have a significant financial impact on {entity_name}.
        - Medium: The topic {topic} presents a medium risk for {entity_name}.
        - Low: The topic {topic} is expected to have a limited impact on {entity_name}.
        - Neutral: The topic {topic} will not affect {entity_name}. 
    2. Provide a one-sentence explanation of the risk of {topic} and its potential financial impact on {entity_name}, across all reports.  
    Important: You should focus only on elements of the input text that explicitly mention direct implication on {entity_name}'s business and ignore any predictions on the general economy or other businesses. Only use the information present in the text. Do NOT infer from outside knowledge.
    If {entity_name} is making predictions on the general economy, is analyzing other companies or comments other businesses, you should score it as "Neutral".
    
    Output:
    Return the result as a JSON object with the following structure:
    {{"magnitude": "High" | "Medium" | "Low" | "Neutral", "summary": "Your summary here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Asynchronous request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )

    risk_magnitude_dict = response.model_dump()

    try:
        risk_magnitude_dict = eval(risk_magnitude_dict['choices'][0]['message']['content'])
    except:
        risk_magnitude_dict = {"magnitude": "Error", "summary": "Error"}  # Default in case of error
    
    return risk_magnitude_dict


def topic_summary_by_company(text_to_analyze, model, number_of_reports, api_key, topic, entity_name):
    # Create the prompt incorporating the topic
    prompt = (
        f"""You are an expert analyst tasked with summarizing the key aspects of {topic} related to {entity_name} based on a set of daily news reports.

        Your goal is to analyze the provided text and summarize the key elements related to {entity_name} in the context of {topic}.

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
        - **Summarize the text:**  Provide a **one-sentence, synthetic summary** of the aspects of {topic} and how it affects {entity_name} across all reports. 
        - Important: You should focus only on elements of the input text that explicitly mention direct implication on {entity_name}'s business. You should ignore any predictions on the general economy or other businesses. Only use the information present in the text. Do NOT infer from outside knowledge.
        - Reference specific dates whenever possible, accurately report names, dates and figures mentioned in the text.
    
    Output:
    **Summary of the key aspects:** A summary of the key aspects related to {topic} that affect {entity_name}.

    Format:
    Return the result as a JSON object with the following structure:
    {{ "summary": "Your summary here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    try:
        summary = eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary = ''
    
    return summary





def topic_summary(text_to_analyze, model, number_of_reports, api_key, main_theme, topic):
    # Create the prompt incorporating the topic
    prompt = (
        f""""You are an expert analyst tasked with summarizing the key aspects of {topic} related to {main_theme} based on a set of news reports.

        Your goal is to analyze the provided text and summarize the key elements related to {topic} in the context of {main_theme}.

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
        - **Summarize the text:**  Provide a **short, synthetic summary** of the key mentioned aspects of {topic} in the context of {main_theme} across all reports. The summary should have between 1 and 3 sentences.
        - Reference specific dates whenever possible, accurately report names, dates and figures mentioned in the text.

    Important: Only use the information present in the text. Do NOT infer from outside knowledge.
    
    Output:
    **Summary of the key aspects:** A summary of the key aspects related to {topic} in the context of {main_theme}

    Format:
    Return the result as a JSON object with the following structure:
    {{ "summary": "Your summary here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    try:
        summary = eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary = ''
    
    return summary
