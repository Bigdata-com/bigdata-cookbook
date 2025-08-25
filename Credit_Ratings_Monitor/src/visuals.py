import os
from IPython.display import display, HTML
import re
import plotly.graph_objects as go

class ReportVisualizer():
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the ReportVisualizer with output directory configuration.

        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = output_dir
        self.FITCH_RATINGS = {
            1: "D",
            2: "RD",
            3: "C",
            4: "CC",
            5: "CCC-",
            6: "CCC",
            7: "CCC+",
            8: "B-",
            9: "B",
            10: "B+",
            11: "BB-",
            12: "BB",
            13: "BB+",
            14: "BBB-",
            15: "BBB",
            16: "BBB+",
            17: "A-",
            18: "A",
            19: "A+",
            20: "AA-",
            21: "AA",
            22: "AA+",
            23: "AAA"}

        self.SP_RATINGS = {
            1: "D",
            2: "C",
            3: "CC",
            4: "CCC-",
            5: "CCC",
            6: "CCC+",
            7: "B-",
            8: "B",
            9: "B+",
            10: "BB-",
            11: "BB",
            12: "BB+",
            13: "BBB-",
            14: "BBB",
            15: "BBB+",
            16: "A-",
            17: "A",
            18: "A+",
            19: "AA-",
            20: "AA",
            21: "AA+",
            22: "AAA"
        }

        self.MOODYS_RATINGS = {
            1: "C",
            2: "Ca",
            3: "Caa3",
            4: "Caa2",
            5: "Caa1",
            6: "B3",
            7: "B2",
            8: "B1",
            9: "Ba3",
            10: "Ba2",
            11: "Ba1",
            12: "Baa3",
            13: "Baa2",
            14: "Baa1",
            15: "A3",
            16: "A2",
            17: "A1",
            18: "Aa3",
            19: "Aa2",
            20: "Aa1",
            21: "Aaa"
        }

        # Invert the rating mappings for each agency
        self.FITCH_RATINGS_INV = {v: k for k, v in self.FITCH_RATINGS.items()}
        self.SP_RATINGS_INV = {v: k for k, v in self.SP_RATINGS.items()}
        self.MOODYS_RATINGS_INV = {v: k for k, v in self.MOODYS_RATINGS.items()}

    # Function to add line breaks
    def add_line_breaks(self, text, max_len=50):
        
        if isinstance(text, list):
            text = "; ".join(text)
        
        if type(text)==str:
            lines = []
            words = text.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > max_len:
                    lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " "
                    current_line += word
            lines.append(current_line)
            return "<br>".join(lines)
        else:
            return None
        
    def convert_text_to_html(self, entity_name, start_date, end_date, text):
        # Define a darker color that works well on both light and dark backgrounds
        universal_text_color = "#7a7a7a"  # Darker gray for better readability
        
        # Title and subtitle
        title_html = f"""
        <div style="text-align: center; font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto;">
            <h1 style="color: #00bfff; font-size: 28px; margin-bottom: 0.5em;">{entity_name}</h1>
            <h3 style="color: #aaa; font-size: 18px; margin-top: 0;">{start_date} - {end_date}</h3>
        </div>
        """
        
        # Escape $ symbols to avoid LaTeX interpretation
        text = text.replace('$', r'\$')

        # Convert main section headers
        text = re.sub(r'### (.+)', r'<h2 style="color: #00bfff; font-size: 24px; margin-top: 1.5em;">\1</h2>', text)

        # Convert subsection headers under "Comprehensive Report"
        text = re.sub(r'\*\*(Credit Ratings|Credit Outlooks|Company|Additional Details)\*\*:', r'<h3 style="color: #00bfff; font-size: 20px; margin-top: 1.2em;">\1</h3>', text)

        # Remove any remaining ** characters used for bold emphasis
        text = re.sub(r'\*\*', '', text)

        # Convert `[source name](url)` into clickable hyperlinks or plain text if URL is 'None'
        def replace_links(match):
            source_name = match.group(1)
            url = match.group(2)
            if url.strip().lower() == 'none':  # Handle cases where URL is 'None'
                return source_name
            return f'<a href="{url}" style="color: #00bfff;" target="_blank">{source_name}</a>'

        text = re.sub(r'\[([^\]]+)\]\((.*?)\)', replace_links, text)

        # Process each line that starts with a date and convert it into an <li> tag with the date in bold
        lines = text.splitlines()
        processed_lines = []
        in_list = False

        for line in lines:
            # Match lines starting with a date pattern (e.g., YYYY-MM-DD)
            if re.match(r'^\s*-\s\d{4}-\d{2}-\d{2}:', line):
                if not in_list:
                    processed_lines.append("<ul>")  # Start a new list
                    in_list = True
                # Format the line as a list item
                line = re.sub(r'^\s*-\s', '', line)  # Remove leading dash
                line = re.sub(r'(\d{4}-\d{2}-\d{2})', r'<strong>\1</strong>', line)  # Bold the date
                line = re.sub(r' - ', r'<br>', line)  # Replace ' - ' with line breaks within the same item
                line = f"<li style='color: {universal_text_color}; line-height: 1.6;'>{line.strip()}</li>"
            else:
                # Close the list if we encounter a non-list item
                if in_list:
                    processed_lines.append("</ul>")
                    in_list = False
                # Process other lines normally
                line = line.strip()
            
            processed_lines.append(line)
        
        # Ensure list is closed if it was left open
        if in_list:
            processed_lines.append("</ul>")

        # Combine processed lines
        text = "\n".join(processed_lines)

        # Final HTML content with a uniform font size of 16px applied to the entire text block
        html_content = f"""
        {title_html}
        <div style="font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; color: {universal_text_color}; font-size: 16px;">
            {text}
        </div>
        """
        return html_content
        
    def save_html_report(self, html_content, entity_name, start_date, end_date, filename=None):
        """
        Save the HTML content to a file in the output directory
        
        Args:
            html_content: HTML content to save
            entity_name: Name of the entity in the report
            start_date: Start date of the report period
            end_date: End date of the report period
            filename: Optional custom filename. If None, one will be generated
            
        Returns:
            Path to the saved file
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            # Standardize the entity name (lowercase, no spaces, no dots)
            clean_entity = entity_name.lower().replace(' ', '_').replace('.', '')
            # Further clean by replacing any remaining special characters
            clean_entity = re.sub(r'[^a-z0-9_]', '', clean_entity)
            filename = f"{clean_entity}_{start_date}_{end_date}.html"
        if not filename.startswith(entity_name.lower().replace(' ', '_').replace('.', '')):
            # Standardize the entity name (lowercase, no spaces, no dots)
            clean_entity = entity_name.lower().replace(' ', '_').replace('.', '')
            # Further clean by replacing any remaining special characters
            clean_entity = re.sub(r'[^a-z0-9_]', '', clean_entity)
            # Add entity name prefix to the filename
            filename = f"{clean_entity}_{filename}"

        # Ensure the filename has .html extension
        if not filename.endswith('.html'):
            filename += '.html'
            
        # Full path
        file_path = os.path.join(self.output_dir, filename)
        
        # Write the HTML content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTML report saved to: {file_path}")
        return file_path
    
    def display_html_report(self, html_content):
        """
        Display HTML content in the notebook
        
        Args:
            html_content: HTML content to display
        """
        display(HTML(html_content))

    def map_rating_and_standardize(self, row):
        rating = row['Credit Rating']
        rater = row['Rater']
        
        # Determine the numeric rating and standardize the agency name
        if "S&P" in rater:
            numeric_rating = self.SP_RATINGS_INV.get(rating, None)
            rating_agency = "S&P Global Inc."
        elif "Moody" in rater:
            numeric_rating = self.MOODYS_RATINGS_INV.get(rating, None)
            rating_agency = "Moody's Corp."
        elif "Fitch" in rater:
            numeric_rating = self.FITCH_RATINGS_INV.get(rating, None)
            rating_agency = "Fitch Ratings Inc."
        else:
            numeric_rating = None
            rating_agency = None  # For unrecognized raters

        return numeric_rating, rating_agency

    def plot_credit_ratings(self, df, save_path=None):
        """
        Plot credit ratings over time and optionally save the figure
        
        Args:
            df: DataFrame containing credit ratings data
            save_path: Optional path to save the figure. If None, the figure is not saved
                       Can be a full path or just a filename (which will be saved in output_dir)
        
        Returns:
            Plotly figure object
        """
        ratee = df['Ratee Entity'].unique()[0]
        
        # Apply the function to get both numeric rating and standardized agency
        df[['numeric_rating', 'standardized_agency']] = df.apply(
            self.map_rating_and_standardize, axis=1, result_type="expand"
        )
        
        # Use standardized agency names to ensure consistent grouping
        unique_agencies = df['standardized_agency'].dropna().unique()

        # Calculate dynamic separation factor based on the range of ratings
        agency_ranges = {
            agency: (df[df['standardized_agency'] == agency]['numeric_rating'].min(),
                    df[df['standardized_agency'] == agency]['numeric_rating'].max())
            for agency in unique_agencies if agency is not None
        }

        # Calculate the maximum range of any agency
        if agency_ranges:
            max_rating_range = max((max_range - min_range) for min_range, max_range in agency_ranges.values() if None not in (min_range, max_range))
            buffer = 2  # Increased buffer to avoid text/marker collisions
        else:
            max_rating_range = 0
            buffer = 2

        # Ensure enough vertical separation for the most extreme case
        separation_factor = max_rating_range + buffer

        # Define offsets for each agency to assign them separate vertical ranges - use standardized names
        agency_offsets = {agency: i * separation_factor * 2 for i, agency in enumerate(unique_agencies)}

        # Initialize the plotly figure
        fig = go.Figure()

        # Iterate through each rating agency and add its line to the plot
        for agency in unique_agencies:
            df_agency = df[df['standardized_agency'] == agency]

            # Apply the offset to numeric_rating to create separate ranges
            df_agency['numeric_rating_offset'] = df_agency['numeric_rating'] + agency_offsets[agency]

            # Add a line trace for the current agency
            fig.add_trace(go.Scatter(
                x=df_agency['Date'],
                y=df_agency['numeric_rating_offset'],
                mode='lines+markers+text',
                name=agency,  # Use standardized agency name
                text=df_agency['Credit Rating'],  # Show the credit rating next to points
                hoverinfo='text+name',
                hovertext=df_agency.apply(
                    lambda row: f"Date: {row['Date']}<br>Rater: {row['Rater']}<br>Rating: {row['Credit Rating']}<br>Key Driver: {self.add_line_breaks(row['Key Driver'])}",
                    axis=1
                ),
                textposition='top center',
                marker=dict(size=10)
            ))

        # Set the overall y-axis range to encompass all agency ranges
        if agency_offsets:
            total_offset = max(agency_offsets.values()) + separation_factor
            min_y = df['numeric_rating'].min() - 0.5
            max_y = df['numeric_rating'].max() + total_offset
        else:
            min_y = 0
            max_y = 25  # Default range if no agencies

        # Customize the layout
        fig.update_layout(
            title=f"Credit Ratings Over Time for {ratee}",
            xaxis=dict(title='Date', automargin=True),
            yaxis=dict(
                title=None,  # Remove axis title
                range=[min_y, max_y],  # Set custom range
                showticklabels=False  # Remove tick labels
            ),
            margin=dict(t=80, r=50, l=50, b=50),  # Add top margin to avoid cutting labels
            template='plotly_white',
            height=600  # Adjust height for better readability
        )

        # Save the figure if a path is provided
        if save_path:
            # Standardize entity name for filename if not already a full path
            if not os.path.dirname(save_path):
                # It's just a filename - check if we should add the entity name
                if not save_path.startswith(ratee.lower().replace(' ', '_').replace('.', '')):
                    # Standardize the entity name (lowercase, no spaces, no dots)
                    clean_entity = ratee.lower().replace(' ', '_').replace('.', '')
                    # Further clean by replacing any remaining special characters
                    clean_entity = re.sub(r'[^a-z0-9_]', '', clean_entity)
                    # Add entity name prefix to the filename
                    save_path = f"{clean_entity}_{save_path}"
                
                # It's just a filename, prepend output_dir
                os.makedirs(self.output_dir, exist_ok=True)
                save_path = os.path.join(self.output_dir, save_path)
                
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                
            # Add extension if not present
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html')):
                save_path += '.html'  # Default to HTML for interactive plot
                
            # Save the figure in the specified format
            if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                fig.write_image(save_path)
            else:
                fig.write_html(save_path)
                
            print(f"Figure saved to: {save_path}")
            
        # Show the plot
        fig.show()
        
        return fig