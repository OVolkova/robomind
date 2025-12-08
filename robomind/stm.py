# import redis
# import time
#
#
# class ShortTermMemory:
#     def __init__(self):
#         # Connect to Redis
#         self.redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
#
#     def add_entry(self, text, role):
#         # Function to add an entry to the "table"
#         if role not in ["agent", "human"]:
#             raise ValueError("Role must be either 'agent' or 'human'")
#
#         entry = {
#             "text": text,
#             "role": role
#         }
#         self.redis_client.hset(str(int(time.time())),mapping=entry)
#
#     def get_entry(self, key):
#         # Function to retrieve an entry by key
#         return self.redis_client.hgetall(key)
#
#     def get_all_entries(self):
#         # Function to get all entries
#         keys = self.redis_client.keys()
#         return [(key, self.redis_client.hgetall(key)) for key in keys]
#
#
# if __name__ == "__main__":
#     # Example usage
#     stm = ShortTermMemory()
#
#     # Add entries
#     stm.add_entry("Hello, how can I help you?", "agent")
#     stm.add_entry("I need assistance with my account.", "human")
#
#     # Retrieve and print all entries
#     all_entries = stm.get_all_entries()
#     for k, e in all_entries:
#         print(f"Key: {k}, Entry: {e}")
