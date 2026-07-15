import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import Book from './models/Book.js';

const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/library-management';

app.use(cors());
app.use(express.json());

app.get('/api/books', async (req, res) => {
  try {
    const books = await Book.find().sort({ createdAt: -1 });
    res.json(books);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch books', error: error.message });
  }
});

app.post('/api/books', async (req, res) => {
  const { title, author, genre } = req.body;
  if (!title || !author || !genre) return res.status(400).json({ message: 'All fields are required' });

  try {
    const book = await Book.create({ title, author, genre, available: true, borrowedBy: '' });
    res.status(201).json(book);
  } catch (error) {
    res.status(500).json({ message: 'Failed to add book', error: error.message });
  }
});

app.post('/api/books/:id/borrow', async (req, res) => {
  try {
    const book = await Book.findById(req.params.id);
    if (!book) return res.status(404).json({ message: 'Book not found' });
    if (!book.available) return res.status(400).json({ message: 'Book already borrowed' });

    book.available = false;
    book.borrowedBy = req.body.borrower || 'Unknown';
    await book.save();
    res.json(book);
  } catch (error) {
    res.status(500).json({ message: 'Failed to borrow book', error: error.message });
  }
});

app.post('/api/books/:id/return', async (req, res) => {
  try {
    const book = await Book.findById(req.params.id);
    if (!book) return res.status(404).json({ message: 'Book not found' });
    if (book.available) return res.status(400).json({ message: 'Book is already available' });

    book.available = true;
    book.borrowedBy = '';
    await book.save();
    res.json(book);
  } catch (error) {
    res.status(500).json({ message: 'Failed to return book', error: error.message });
  }
});

app.delete('/api/books/:id', async (req, res) => {
  try {
    const deleted = await Book.findByIdAndDelete(req.params.id);
    if (!deleted) return res.status(404).json({ message: 'Book not found' });
    res.json({ message: 'Book deleted' });
  } catch (error) {
    res.status(500).json({ message: 'Failed to delete book', error: error.message });
  }
});

const startServer = (port = PORT) => {
  const server = app.listen(port, () => {
    console.log(`Library API running on http://localhost:${port}`);
  });

  server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
      const nextPort = port + 1;
      console.warn(`Port ${port} is already in use. Trying port ${nextPort}...`);
      if (process.env.PORT) {
        console.error('The configured PORT is in use. Please stop the other process or change PORT in your environment.');
        process.exit(1);
      } else {
        startServer(nextPort);
      }
    } else {
      console.error('Server error:', error.message);
      process.exit(1);
    }
  });
};

mongoose.connect(MONGO_URI, {
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000
})
  .then(() => {
    console.log('Connected to MongoDB');
    startServer();
  })
  .catch((error) => {
    console.error('MongoDB connection error:', error.message);
    console.error('Please confirm MongoDB is running and that the URI is correct.');
    console.error('Try: mongodb://127.0.0.1:27017/library-management');
    process.exit(1);
  });
