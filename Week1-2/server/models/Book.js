import mongoose from 'mongoose';

const bookSchema = new mongoose.Schema(
  {
    title: { type: String, required: true, trim: true },
    author: { type: String, required: true, trim: true },
    genre: { type: String, required: true, trim: true },
    available: { type: Boolean, default: true },
    borrowedBy: { type: String, default: '' }
  },
  { timestamps: true }
);

export default mongoose.model('Book', bookSchema);
