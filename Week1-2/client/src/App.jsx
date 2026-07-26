import { useEffect, useMemo, useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:5000/api/books';

function App() {
  const [books, setBooks] = useState([]);
  const [form, setForm] = useState({ title: '', author: '', genre: '' });
  const [search, setSearch] = useState('');
  const [borrower, setBorrower] = useState('');
  const [genreFilter, setGenreFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');

  const fetchBooks = async () => {
    const res = await fetch(API_URL);
    const data = await res.json();
    setBooks(data);
  };

  useEffect(() => {
    fetchBooks();
  }, []);

  const handleAddBook = async (e) => {
    e.preventDefault();
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form)
    });

    if (res.ok) {
      setForm({ title: '', author: '', genre: '' });
      fetchBooks();
    }
  };

  const handleBorrow = async (id) => {
    const res = await fetch(`${API_URL}/${id}/borrow`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ borrower: borrower || 'Guest' })
    });
    if (res.ok) fetchBooks();
  };

  const handleReturn = async (id) => {
    const res = await fetch(`${API_URL}/${id}/return`, {
      method: 'POST'
    });
    if (res.ok) fetchBooks();
  };

  const handleDelete = async (id) => {
    const res = await fetch(`${API_URL}/${id}`, { method: 'DELETE' });
    if (res.ok) fetchBooks();
  };

  const booksByStatus = useMemo(() => {
    return {
      total: books.length,
      available: books.filter((book) => book.available).length,
      borrowed: books.filter((book) => !book.available).length
    };
  }, [books]);

  const filteredBooks = useMemo(() => {
    return books.filter((book) => {
      const query = search.toLowerCase();
      const matchesSearch =
        book.title.toLowerCase().includes(query) ||
        book.author.toLowerCase().includes(query);

      const matchesGenre = genreFilter === 'all' || book.genre.toLowerCase() === genreFilter;
      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'available' && book.available) ||
        (statusFilter === 'borrowed' && !book.available);

      return matchesSearch && matchesGenre && matchesStatus;
    });
  }, [books, search, genreFilter, statusFilter]);

  const uniqueGenres = useMemo(() => {
    const genres = Array.from(new Set(books.map((book) => book.genre.toLowerCase())));
    return genres.filter((genre) => genre.trim() !== '');
  }, [books]);

  const getBookId = (book) => book._id || book.id;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">📚</div>
          <div className="brand-name">
            <strong>BookNest</strong>
            <span>Library Dashboard</span>
          </div>
        </div>

        <nav className="nav-links">
          <a href="#dashboard">Dashboard</a>
          <a href="#add-book">Add Book</a>
        </nav>
      </header>

      <main className="page-content">
        <section className="hero">
          <div className="hero-text">
            <span className="eyebrow">Library Dashboard</span>
            <h1>Manage and organize your book collection</h1>
            <p>Track availability, borrowings, and add new titles with a polished and responsive management dashboard.</p>
          </div>

          <div className="hero-cta">
            <label>
              Search by title or author
              <input
                type="text"
                placeholder="Start typing to search books"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>
            <label>
              Selected Borrower Name
              <input
                type="text"
                placeholder="Guest"
                value={borrower}
                onChange={(e) => setBorrower(e.target.value)}
              />
            </label>
            <p>Use one search input and clear filters to quickly find books in your catalog.</p>
          </div>
        </section>

        <section className="stats-grid" id="dashboard">
          <article className="stat-card">
            <div className="stat-icon">📖</div>
            <h3>Total Books</h3>
            <p>{booksByStatus.total}</p>
          </article>
          <article className="stat-card">
            <div className="stat-icon">✅</div>
            <h3>Available Books</h3>
            <p>{booksByStatus.available}</p>
          </article>
          <article className="stat-card">
            <div className="stat-icon">🔖</div>
            <h3>Borrowed Books</h3>
            <p>{booksByStatus.borrowed}</p>
          </article>
        </section>

        <section className="card-panel" id="add-book">
          <div className="panel-header">
            <h2>Add New Book</h2>
          </div>
          <form className="book-form" onSubmit={handleAddBook}>
            <label>
              Title
              <input
                type="text"
                placeholder="Enter book title"
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
                required
              />
            </label>
            <label>
              Author
              <input
                type="text"
                placeholder="Enter author name"
                value={form.author}
                onChange={(e) => setForm({ ...form, author: e.target.value })}
                required
              />
            </label>
            <label>
              Genre
              <input
                type="text"
                placeholder="Enter book genre"
                value={form.genre}
                onChange={(e) => setForm({ ...form, genre: e.target.value })}
                required
              />
            </label>
            <button type="submit">+ Add Book</button>
          </form>
        </section>

        <section className="card-panel">
          <div className="panel-header">
            <h2>Book Collection</h2>
          </div>

          <div className="search-filters">
            <div className="filter-row">
              <label className="search-group">
                Search
                <input
                  type="text"
                  placeholder="Search by title or author"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </label>
            </div>

            <div className="filter-group">
              <label>
                Genre Filter
                <select value={genreFilter} onChange={(e) => setGenreFilter(e.target.value)}>
                  <option value="all">All Genres</option>
                  {uniqueGenres.map((genre) => (
                    <option key={genre} value={genre}>{genre.charAt(0).toUpperCase() + genre.slice(1)}</option>
                  ))}
                </select>
              </label>
              <label>
                Status Filter
                <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                  <option value="all">All Books</option>
                  <option value="available">Available</option>
                  <option value="borrowed">Borrowed</option>
                </select>
              </label>
            </div>
          </div>

          {filteredBooks.length === 0 ? (
            <div className="empty-state">
              <h3>No books found</h3>
              <p>Try adjusting your search or filters to locate books in the collection.</p>
            </div>
          ) : (
            <div className="cards-grid">
              {filteredBooks.map((book) => {
                const bookId = getBookId(book);
                return (
                  <article key={bookId} className="book-card">
                    <div className="book-card-header">
                      <div className="book-icon">📘</div>
                      <div>
                        <h3 className="book-card-title">{book.title}</h3>
                        <p className="book-card-meta">{book.author}</p>
                      </div>
                      <span className={`status-chip ${book.available ? 'available' : 'borrowed'}`}>
                        {book.available ? 'Available' : 'Borrowed'}
                      </span>
                    </div>

                    <div className="card-details">
                      <div className="detail-row">
                        <span>Genre</span>
                        <strong>{book.genre}</strong>
                      </div>
                      {!book.available && (
                        <div className="detail-row">
                          <span>Borrowed by</span>
                          <strong>{book.borrowedBy || 'Guest'}</strong>
                        </div>
                      )}
                    </div>

                    <div className="card-actions">
                      {book.available ? (
                        <button className="action-btn" onClick={() => handleBorrow(bookId)}>
                          Borrow
                        </button>
                      ) : (
                        <button className="action-btn" onClick={() => handleReturn(bookId)}>
                          Return
                        </button>
                      )}
                      <button className="delete-btn" onClick={() => handleDelete(bookId)} title="Delete book">
                        ✕
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
