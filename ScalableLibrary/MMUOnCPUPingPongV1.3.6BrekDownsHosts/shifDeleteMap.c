#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 256

typedef struct {
    void* key;
    int value;
} KeyValuePair;

typedef struct {
    KeyValuePair pairs[MAX_SIZE];
    int size;
} KeyValueStore;

void initializeStore(KeyValueStore* store) {
    store->size = 0;
}

int insertPair(KeyValueStore* store, void* key, int value) {
    if (store->size >= MAX_SIZE) {
        printf("Store is full.\n");
        return -1; // Failure
    }

    // Check for existing key
    for (int i = 0; i < store->size; i++) {
        if (store->pairs[i].key == key) {
            printf("Key already exists.\n");
            return -2; // Failure
        }
    }

    // Insert new key-value pair
    store->pairs[store->size].key = key;
    store->pairs[store->size].value = value;
    store->size++;
    return 0; // Success
}

int deletePair(KeyValueStore* store, void* key) {
    for (int i = 0; i < store->size; i++) {
        if (store->pairs[i].key == key) {
            // Found key, now delete it by shifting the rest of the array
            for (int j = i; j < store->size - 1; j++) {
                store->pairs[j] = store->pairs[j + 1];
            }
            store->size--;
            return 0; // Success
        }
    }
    printf("Key not found.\n");
    return -1; // Failure
}

int findValueByKey(const KeyValueStore* store, void* key) {
    for (int i = 0; i < store->size; i++) {
        if (store->pairs[i].key == key) {
            return store->pairs[i].value;
        }
    }
    printf("Key not found.\n");
    return -1; // Indicates failure or key not found
}

int main() {
    KeyValueStore store;
    initializeStore(&store);

    // Example usage
    int keys[5];
    for (int i = 0; i < 5; i++) {
        keys[i] = i;
        insertPair(&store, &keys[i], i * 100);
    }

    // Find value by key
    int value = findValueByKey(&store, &keys[2]);
    if (value != -1) {
        printf("Value found: %d\n", value);
    }

    // Delete a pair
    deletePair(&store, &keys[2]);
    value = findValueByKey(&store, &keys[2]);
    if (value == -1) {
        printf("Key successfully deleted.\n");
    }

    return 0;
}
