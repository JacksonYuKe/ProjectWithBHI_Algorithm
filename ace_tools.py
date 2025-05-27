"""
Simple tools for displaying dataframes to the user.
"""

def display_dataframe_to_user(name, dataframe):
    """
    Display a dataframe to the user with a name/title.
    
    Args:
        name (str): The name or title for the dataframe
        dataframe (pandas.DataFrame): The dataframe to display
    """
    print(f"\n--- {name} ---")
    print(f"Shape: {dataframe.shape}")
    print(dataframe.head())
    print("...")
