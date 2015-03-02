## Class 17 - Databases and Map/Reduce

### Databases

- Organized collections of data
- Organized using a schema
- Schema represents different tables (often)

Why use one?

- Easy to store and retreive data
- Generally has a structured language for interacting with the data
- Reliable and scalable
- Access large amounts of data relatively quickly

Databases can be visualized using UML (Unified Modeling Language)

- See ULM 2.0 book for more depth on this

#### Relational DBs

Currently the industry standard.

- Traditional row/column structure
- **Strict** structure / Primary Keys
- Entire column for each feature
- Examples: MySQL, Oracle, Postgres, SQLite

#### NoSQL

More popular among startups than in established industry.

- No well-defined data structure
- Works better for unstructured data
- Cheaper hardware
- Examples: MongoDB, CouchDB, Redis, Cassandra

### SQL

Example in code, using sqlite3 in Python:

```python

# python package to interface with database files
import sqlite3 as lite

# connect to a local database
con = lite.connect('data/tweets.db')

# create a Cursor object -- implements execution of stuff in db
cur = con.cursor()

# select everything from tweets
cur.execute('SELECT * from Tweets')
cur.fetchall()

# insert a record
cur.execute("INSERT INTO Tweets VALUES(9,'Investors are claiming that $TSLA will only continue to fall!', -.67)")
# need t0 commit to make sure that all changes are done
# NB this is called on the connection, not the cursor
con.commit()

# select only a few columns
cur.execute('SELECT Text, Id from Tweets')
cur.fetchall()

# select with an id
cur.execute('SELECT Text, Sentiment from Tweets WHERE Id = 5')
cur.fetchone()  # returns a tuple, rather than a list containing one tuple

# grab all tweets with negative sentiment
cur.execute('SELECT Text, Sentiment from Tweets WHERE Sentiment  < 0')
cur.fetchall()


# close the connection if we are done with it.
con.close()

```

### Map/Reduce

Stepwise process of map / reduce, in functional programming:

1. Map: an operation that takes a function mapping a domain D onto a range R, and a list of elements of D
2. Filter: an operation that takes a function mapping some domain D onto the range [True, False], and a list of elements from D, and produces a subset of this input list
3. Reduce: an operation that takes an operator, and a list, and produces the single value resulting from the operator being applied recursively to the first and second elements of the list, with the result of the operation replacing those elements

From slides, for MapReduce:

1. Map: produces key-value pairs depending on particular task
2. Shuffle/Sort: sorts the key-value pairs; could make new ones
3. Reduce: combines the key-value pairs into a single output

Ideally, the map stage is run over a cluster of computers in parallel.

Benefits:

- Very scalable across different data sizes
- Many implementations w/ documentation exist for popular programming languages
- Relatively fast compared to straight-through processing

Helpful differentiated definition from Wikipedia:

**MapReduce** is a programming model and an associated implementation for processing and generating large data sets with a parallel, distributed algorithm on a cluster.

A MapReduce program is composed of a **Map()** procedure that performs filtering and sorting (such as sorting students by first name into queues, one queue for each name) and a **Reduce()** procedure that performs a summary operation (such as counting the number of students in each queue, yielding name frequencies). The "MapReduce System" (also called "infrastructure" or "framework") orchestrates the processing by marshalling the distributed servers, running the various tasks in parallel, managing all communications and data transfers between the various parts of the system, and providing for redundancy and fault tolerance.

The model is inspired by the map and reduce functions commonly used in functional programming,[3] although their purpose in the MapReduce framework is not the same as in their original forms.[4] The key contributions of the MapReduce framework are not the actual map and reduce functions, but the scalability and fault-tolerance achieved for a variety of applications by optimizing the execution engine once. As such, a single-threaded implementation of MapReduce (such as MongoDB) will usually not be faster than a traditional (non-MapReduce) implementation, any gains are usually only seen with multi-threaded implementations.[5] Only when the optimized distributed shuffle operation (which reduces network communication cost) and fault tolerance features of the MapReduce framework come into play, is the use of this model beneficial. Optimizing the communication cost is essential to a good MapReduce algorithm.[6]

MapReduce libraries have been written in many programming languages, with different levels of optimization. A popular open-source implementation that has support for distributed shuffles is part of Apache Hadoop. The name MapReduce originally referred to the proprietary Google technology, but has since been genericized.
