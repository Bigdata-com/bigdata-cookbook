import tiktoken
from typing import List, Optional
import logging

class TokenManager:
    """
    Manages token counting and text splitting for large language model API calls.
    Handles text chunking to fit within model token limits while preserving context.
    
    Attributes:
        encoder (tiktoken.Encoding): TikToken encoder for the specified model
        max_tokens (int): Maximum tokens allowed in a single API call
        timeout (int): Maximum time in seconds to wait for API response
        verbose (bool): Whether to print detailed processing information
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini-2024-07-18", 
                 max_tokens: int = 4000,
                 timeout: int = 30,
                 verbose: bool = False):
        """
        Initialize the TokenManager with model-specific settings.
        
        Args:
            model_name (str): Name of the language model being used
            max_tokens (int): Maximum tokens per API call
            timeout (int): API timeout in seconds
            verbose (bool): Enable detailed logging
        """
        # Initialize the tokenizer for the specified model
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.verbose = verbose
        
        # Set up logging if verbose mode is enabled
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text string.
        
        Args:
            text (str): Input text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    
    def split_text_by_chunks(self, df_text_to_summarize) -> List[str]:
        """
        Split dataframe into concatanated chunks that fit within the token limit.
        Iterate over rows in the dataframe.
        Preserve chunks.
        
        Args:
            df_text_to_summarize (DataFrame): Input text to split into chunks
            
        Returns:
            List[str]: List of text chunks within token limit
        """

        tokens_count = 0
        list_chunks = []
        list_concatanated_chunks = []
        
        for _, df_row in df_text_to_summarize.iterrows():
            chunk = df_row['text_to_summarize']
            chunk_tokens = self.count_tokens(chunk)

            if (tokens_count+chunk_tokens)<self.max_tokens:             
                list_chunks.append(chunk)
                tokens_count += chunk_tokens
            else:
                if self.verbose:
                    self.logger.info(f"Initial token count: {tokens_count} tokens (limit: {self.max_tokens})")
                concat_chunk = "\n\n".join(list_chunks)
                list_concatanated_chunks.append(concat_chunk)
                list_chunks = [chunk]
                tokens_count = chunk_tokens

        if list_chunks:
            concat_chunk = "\n\n".join(list_chunks)
            list_concatanated_chunks.append(concat_chunk)
            
        return list_concatanated_chunks

        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within the token limit.
        Preserves sentence boundaries where possible.
        
        Args:
            text (str): Input text to split into chunks
            
        Returns:
            List[str]: List of text chunks within token limit
        """
        if not text:
            return []
        
        # Log initial text statistics if verbose mode is enabled    
        if self.verbose:
            self.logger.info(f"\nProcessing text of length {len(text)} characters...")
            initial_tokens = self.count_tokens(text)
            self.logger.info(f"Initial token count: {initial_tokens} tokens (limit: {self.max_tokens})")
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split text into sentences, preserving the period
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Handle sentences that exceed token limit
            if sentence_tokens > self.max_tokens:
                # If a single sentence is too long, split it into smaller pieces
                words = sentence.split()
                current_words = []
                current_word_length = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word)
                    
                    # Check if adding word would exceed limit
                    if current_word_length + word_tokens > self.max_tokens:
                        if current_words:
                            # Create chunk from accumulated words
                            chunk_text = ' '.join(current_words) + '.'
                            chunks.append(chunk_text)
                            if self.verbose:
                                self.logger.info(
                                    f"Created chunk with {self.count_tokens(chunk_text)} tokens"
                                )
                                
                        # Start new chunk with current word
                        current_words = [word]
                        current_word_length = word_tokens
                    else:
                        # Add word to current chunk
                        current_words.append(word)
                        current_word_length += word_tokens
                
                # Handle any remaining words
                if current_words:
                    chunk_text = ' '.join(current_words) + '.'
                    chunks.append(chunk_text)
                    if self.verbose:
                        self.logger.info(
                            f"Created chunk with {self.count_tokens(chunk_text)} tokens"
                        )
                    
            # Handle normal sentences
            elif current_length + sentence_tokens > self.max_tokens:
                # Current chunk would exceed limit, start new chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk) + '.'
                    chunks.append(chunk_text)
                    if self.verbose:
                        self.logger.info(
                            f"Created chunk with {self.count_tokens(chunk_text)} tokens"
                        )
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add the final chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk) + '.'
            chunks.append(chunk_text)
            if self.verbose:
                self.logger.info(
                    f"Created chunk with {self.count_tokens(chunk_text)} tokens"
                )
        
        # Log final chunking results if verbose mode is enabled
        if self.verbose:
            self.logger.info(f"Split text into {len(chunks)} chunks")
            
        return chunks
    
    def validate_text(self, text: str) -> Optional[str]:
        """
        Validate text fits within token limit or can be split appropriately.
        
        Args:
            text (str): Input text to validate
            
        Returns:
            Optional[str]: Validated text or None if invalid
            
        Raises:
            ValueError: If text cannot be processed within token limits
        """
        if not text:
            return None
            
        token_count = self.count_tokens(text)
        
        # Text fits within limit
        if token_count <= self.max_tokens:
            return text
            
        # Check if text can be split
        chunks = self.split_text(text)
        if not chunks:
            raise ValueError(
                f"Text cannot be processed: Token count {token_count} exceeds limit "
                f"and text cannot be split into smaller chunks"
            )
            
        # Return first chunk if splitting is required
        return chunks[0]
