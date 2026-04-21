import ChatInput from "@/components/ChatInput";

function App() {

  return (
    <div className="p-4 flex flex-col items-center justify-center ">
       <h1 className="text-4xl font-bold mt-96 mb-4">Welcome to the AI Frontend</h1>
        <div className="w-2/3">
          <ChatInput />
        </div>
    </div>
  );
}

export default App
