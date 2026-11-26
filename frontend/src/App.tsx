import { useEffect, useState } from "react";
import { Chat } from "./components/Chat";
import { AuthModal } from "./components/AuthModal";

function App() {
  // null while the server is being asked; the access prompt shows only once it says no
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/me")
      .then((response) => response.json())
      .then((me) => setIsAuthenticated(me.signed_in === true))
      .catch(() => setIsAuthenticated(false));
  }, []);

  return (
    <>
      {isAuthenticated === false && (
        <AuthModal onAuthenticated={() => setIsAuthenticated(true)} />
      )}
      <Chat onUnauthorized={() => setIsAuthenticated(false)} />
    </>
  );
}

export default App;
