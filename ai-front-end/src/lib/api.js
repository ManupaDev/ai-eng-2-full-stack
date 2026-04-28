export const sendMessage = async (message) => {
    try {
       const newMessages = await fetch("http://localhost:8000/api/messages", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: message }),
        });
        const data = await newMessages.json();
        return data.messages;
    } catch (error) {
        console.error("Error sending message:", error);        
    }
}


export const getMessages = async () => {
    try {
        const res = await fetch("http://localhost:8000/api/messages", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        const data = await res.json();
        return data.messages;
    } catch (error) {
        console.error("Error fetching messages:", error);
        return [];
    }
}