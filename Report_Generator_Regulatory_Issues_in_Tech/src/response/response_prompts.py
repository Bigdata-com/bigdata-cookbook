from openai import OpenAI


def topic_condense_by_company(text_to_analyze, model, api_key, company_name):
    # Create the prompt
    prompt = (
    f"""You are an expert assisting a search engine tool that uses similarity search to retrieve sentences related to regulatory issues in the Tech sector.

    Your goal is to summarize the provided text in a short sentence that will be used to query a search engine tool. We want to retrieve documents related to the company's response to the issue described in the provided text.
    
    Output:
    **Short sentence:** Your short sentence.

    Format:
    Return the result as a JSON object with the following structure:
    {{ "sentence": "Your short sentence here"}}
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
        summary = eval(summary['choices'][0]['message']['content'])['sentence']
    except:
        summary = ''
    
    return summary


def response_summary_by_company(text_to_analyze, model, number_of_reports, api_key, topic, topic_summary, entity_name):
    # Create the prompt incorporating the topic
    prompt = (
        f"""You are an expert analyst tasked with analyzing the response of {entity_name} to {topic} issues based on a set of company's filings and transcripts or news.

    Topic Summary:
    Read carefulfy the following topic summary that describes the issues that {entity_name} faces regarding {topic}: {topic_summary}

    Goal:
    Your goal is to analyze the provided text and summarize the key elements related to {entity_name}'s response to the issue discribed in the topic summary and related to {topic}. 
    Extract from the text the relevant details (e.g. company names, product names, project names...) and facts (e.g. lawsuits, regulation names...) that illustrate this response. Only extract facts that are present in the provided text.
    You should focus on the company's response and mitigation plan and avoid repeating what is in the topic summary. Only mention elements that will be applied to {entity_name} and ignore any recommandation to other companies. 

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
        - **Summarize the text:**  Provide a **one-sentence, synthetic summary** of the aspects of {entity_name}'s response to the issue related to {topic} and discribed in the topic summary, across all reports. Please ensure your summary focuses on how {entity_name} plans to address these issues. When possible, your summary should extract from the text the relevant details and facts that illustrate this response. 
        - Focus on company's response, and don't repeat what is in the Topic Summary. 
        - Reference specific dates whenever possible, accurately report names, dates and figures mentioned in the text.
    
    Output:
    **Summary of the company's response:** A summary of the key aspects related to {entity_name}'s response to {topic}.

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


def old_response_summary_by_company(text_to_analyze, model, number_of_reports, api_key, topic, topic_summary, entity_name):
    # Create the prompt incorporating the topic
    prompt = (
        f"""You are an expert analyst tasked with analyzing the response of {entity_name} to {topic} issues based on a set of company's filings and transcripts or news.

        Your goal is to analyze the provided text and summarize the key elements related to {entity_name}'s response to the issue discribed in {topic_summary} and related to {topic}. 
        Extract from the text the relevant details (e.g. company names, product names, project names...) and facts (e.g. lawsuits, regulation names...) that illustrate this response. Only extract facts that are present in the provided text.

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
    **Summarize the text:**  Provide a **one-sentence, synthetic summary** of the aspects of {entity_name}'s response to the issue related to {topic} and discribed in {topic_summary}, across all reports. Please ensure your summary focuses on how {entity_name} plans to address these issues. When possible, your summary should extract from the text the relevant details and facts that illustrate this response.
    
    Output:
    **Summary of the company's response:** A summary of the key aspects related to {entity_name}'s response to {topic}.

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