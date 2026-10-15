// Tiny IndexedDB wrapper. Every answered question is one record in "rounds".
const NAME = "tell-lab";
const STORE = "rounds";
let dbp = null;

function open() {
  if (!dbp) {
    dbp = new Promise((resolve, reject) => {
      const req = indexedDB.open(NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbp;
}

function run(mode, fn) {
  return open().then(db => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const req = fn(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(req ? req.result : undefined);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

export const put = round => run("readwrite", s => { s.put(round); });
export const putMany = rounds => run("readwrite", s => { rounds.forEach(r => s.put(r)); });
export const all = () => run("readonly", s => s.getAll());
export const clear = () => run("readwrite", s => { s.clear(); });
