# decode hex strings for testing 
# also reverse the decoded string for verification
# not all set are strings, there are integers too, so we return both byte array and reversed string
# example: 0x00000010 -> 16 bytes
# example: 0x46464952 -> 'RIFF'



def decode_hex_string(hex_string):
    # Remove '0x' prefix if present
    if hex_string.startswith("0x"):
        hex_string = hex_string[2:]

    # Convert hex string to bytes
    byte_array = bytes.fromhex(hex_string)

    # Try to decode bytes to string, ignoring errors
    try:
        decoded_string = byte_array.decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        decoded_string = ''

    # Reverse the decoded string
    reversed_string = decoded_string[::-1]

    return reversed_string

def decode_hex_to_integer(hex_string):
    # Remove '0x' prefix if present
    if hex_string.startswith("0x"):
        hex_string = hex_string[2:]

    # Convert hex string to integer
    integer_value = int(hex_string, 16)

    return integer_value
    

if __name__ == "__main__":
    # Test cases
    test_hex_strings = [
        "0x46464952",
        "0x00009944", 
        "0x45564157", 
        "0x20746D66", 
        "0x00000010",
        "0x00003E80", 
        "0x00007D00", 
        "0x61746164", 
        "0x00009920",
    ]
    for hex_str in test_hex_strings:
        decoded = decode_hex_string(hex_str)
        print(f"Hex: {hex_str} -> Decoded: '{decoded}'")
        if hex_str == "0x00000010" or hex_str == "0x00003E80" or hex_str == "0x00007D00" or hex_str == "0x00009920" or hex_str == "0x00009944":
            integer_value = decode_hex_to_integer(hex_str)
            print(f"Hex: {hex_str} -> Integer: {integer_value}")